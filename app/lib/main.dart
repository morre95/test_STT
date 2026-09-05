import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

void main() => runApp(const SttLab());

/// One entry from the backend catalog; `languages` differs per model because
/// Vosk ships a separate decoding graph for each language.
class ModelInfo {
  const ModelInfo(this.id, this.name, this.languages, this.available);

  factory ModelInfo.fromJson(Map<String, dynamic> m) => ModelInfo(
      m['id'] as String,
      m['name'] as String,
      (m['languages'] as List).cast<String>(),
      m['available'] == true);

  final String id;
  final String name;
  final List<String> languages;
  final bool available;
}

class SttLab extends StatelessWidget {
  const SttLab({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
      title: 'Local STT Lab',
      theme: ThemeData(
          useMaterial3: true,
          brightness: Brightness.dark,
          scaffoldBackgroundColor: const Color(0xff101417),
          colorScheme: ColorScheme.fromSeed(
              seedColor: const Color(0xffd6f36b), brightness: Brightness.dark)),
      home: const LabPage());
}

class LabPage extends StatefulWidget {
  const LabPage({super.key, this.client});

  /// Injected by the widget tests; the app builds its own client.
  final http.Client? client;

  @override
  State<LabPage> createState() => _LabPageState();
}

class _LabPageState extends State<LabPage> {
  final recorder = AudioRecorder();
  final host = TextEditingController(text: '10.0.2.2:8000');
  late final http.Client client = widget.client ?? http.Client();
  StreamSubscription<Uint8List>? mic;
  WebSocketChannel? socket;
  bool running = false;
  bool loading = false;
  String text = '';
  String status = 'Redo att testa';
  static const connectTimeout = Duration(seconds: 8);
  static const languageNames = {'sv': 'Svenska', 'en': 'English', 'auto': 'Auto'};
  List<ModelInfo> catalog = const [];
  String? model;
  String language = 'sv';
  double? firstMs, finalMs, rtf;

  @override
  void initState() {
    super.initState();
    loadModels();
  }

  String get authority {
    final p = host.text.trim().split(':');
    return '${p[0]}:${p.length > 1 ? p[1] : '8000'}';
  }

  ModelInfo? get selected =>
      catalog.where((m) => m.id == model).firstOrNull;

  /// The catalog is owned by the backend, so the pickers are built from it
  /// rather than from a second list that would drift out of sync.
  Future<void> loadModels() async {
    setState(() => status = 'Hämtar modeller…');
    try {
      final response = await client
          .get(Uri.parse('http://$authority/models'))
          .timeout(connectTimeout);
      if (response.statusCode != 200) {
        throw Exception('HTTP ${response.statusCode}');
      }
      final list = (jsonDecode(utf8.decode(response.bodyBytes)) as List)
          .map((m) => ModelInfo.fromJson(m as Map<String, dynamic>))
          .toList();
      if (!mounted) return;
      setState(() {
        catalog = list;
        status = list.isEmpty ? 'Servern erbjuder inga modeller' : 'Redo att testa';
        _select(catalog.where((m) => m.id == model || m.available).firstOrNull ??
            catalog.firstOrNull);
      });
    } catch (e) {
      debugPrint('Model catalog fetch failed: $e');
      if (!mounted) return;
      setState(() {
        catalog = const [];
        model = null;
        status = 'Kunde inte hämta modeller';
      });
    }
  }

  /// Keeps the language valid for the model: a Vosk checkpoint accepts only its
  /// own language, and the server rejects the run otherwise.
  void _select(ModelInfo? choice) {
    model = choice?.id;
    if (choice != null && !choice.languages.contains(language)) {
      language = choice.languages.first;
    }
  }
  Future<void> _beginMic() async {
    final s = await recorder.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits, sampleRate: 16000, numChannels: 1));
    mic = s.listen((b) {
      for (var i = 0; i < b.length; i += 3200) {
        final end = i + 3200 < b.length ? i + 3200 : b.length;
        socket?.sink.add(Uint8List.sublistView(b, i, end));
      }
    });
    setState(() {
      running = true;
      loading = false;
      status = 'Spelar in…';
    });
  }

  Future<void> start() async {
    if (!await recorder.hasPermission()) {
      setState(() => status = 'Mikrofonbehörighet saknas');
      return;
    }
    final channel =
        WebSocketChannel.connect(Uri.parse('ws://$authority/ws/transcribe'));
    socket = channel;
    setState(() {
      loading = true;
      status = 'Ansluter…';
      text = '';
      firstMs = finalMs = rtf = null;
    });
    try {
      await channel.ready.timeout(connectTimeout);
    } catch (e) {
      debugPrint('WebSocket connect failed: $e');
      await channel.sink.close();
      if (!mounted) return;
      setState(() {
        socket = null;
        status = 'Kunde inte ansluta';
        loading = false;
      });
      return;
    }
    channel.stream.listen((e) async {
      final m = jsonDecode(e);
      if (m['type'] == 'loading') setState(() => status = 'Laddar modell…');
      if (m['type'] == 'ready') await _beginMic();
      if (m['type'] == 'partial') setState(() => text = m['text'] ?? '');
      if (m['type'] == 'error') {
        setState(() {
          status = m['message'] ?? 'Modellfel';
          running = false;
          loading = false;
        });
      }
      if (m['type'] == 'final') {
        setState(() {
          status = 'Klar';
          finalMs = (m['server_finalization_ms'] as num?)?.toDouble();
          firstMs = (m['server_first_text_ms'] as num?)?.toDouble();
          rtf = (m['rtf'] as num?)?.toDouble();
          text = m['text'] ?? text;
          running = false;
          loading = false;
        });
      }
    }, onError: (e) {
      debugPrint('WebSocket stream error: $e');
      setState(() {
        status = 'Anslutningen bröts';
        running = false;
        loading = false;
      });
    });
    channel.sink.add(jsonEncode({
      'type': 'start',
      'model': model,
      'language': language,
      'settings': {}
    }));
  }

  Future<void> stop() async {
    await mic?.cancel();
    await recorder.stop();
    socket?.sink.add(jsonEncode({'type': 'stop'}));
    setState(() => status = 'Bearbetar slutet…');
  }

  @override
  void dispose() {
    mic?.cancel();
    recorder.dispose();
    host.dispose();
    client.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext c) => Scaffold(
      appBar: AppBar(title: const Text('LOCAL STT LAB'), actions: [
        Padding(
            padding: const EdgeInsets.all(16),
            child: Text(status,
                style: TextStyle(
                    color: running ? Colors.orange : const Color(0xffd6f36b))))
      ]),
      body: ListView(padding: const EdgeInsets.all(24), children: [
        Text('Lyssna. Mät. Jämför.',
            style: Theme.of(c)
                .textTheme
                .headlineMedium
                ?.copyWith(fontWeight: FontWeight.w800)),
        const SizedBox(height: 8),
        Text('En lokal arbetsbänk för svenska och engelska talmodeller.',
            style: TextStyle(color: Colors.grey[400])),
        const SizedBox(height: 28),
        TextField(
            controller: host,
            onSubmitted: (_) => loadModels(),
            decoration: InputDecoration(
                labelText: 'FastAPI-server',
                prefixIcon: const Icon(Icons.link),
                suffixIcon: IconButton(
                    tooltip: 'Hämta modeller',
                    onPressed: running || loading ? null : loadModels,
                    icon: const Icon(Icons.refresh)))),
        const SizedBox(height: 16),
        Row(children: [
          Expanded(
              child: DropdownButtonFormField<String>(
                  isExpanded: true,
                  initialValue: model,
                  decoration: const InputDecoration(labelText: 'Modell'),
                  items: [
                    for (final m in catalog)
                      DropdownMenuItem(
                          value: m.id,
                          enabled: m.available,
                          child: Text(
                              m.available ? m.name : '${m.name} (ej installerad)',
                              style: TextStyle(
                                  color: m.available ? null : Colors.grey[600])))
                  ],
                  onChanged: running || loading || catalog.isEmpty
                      ? null
                      : (v) => setState(
                          () => _select(catalog.firstWhere((m) => m.id == v))))),
          const SizedBox(width: 12),
          Expanded(
              child: DropdownButtonFormField<String>(
                  isExpanded: true,
                  initialValue: selected == null ? null : language,
                  decoration: const InputDecoration(labelText: 'Språk'),
                  items: [
                    for (final code in selected?.languages ?? const <String>[])
                      DropdownMenuItem(
                          value: code, child: Text(languageNames[code] ?? code))
                  ],
                  onChanged: running || loading || selected == null
                      ? null
                      : (v) => setState(() => language = v!)))
        ]),
        const SizedBox(height: 24),
        Container(
            padding: const EdgeInsets.all(22),
            decoration: BoxDecoration(
                color: const Color(0xff1a2023),
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: const Color(0xff303a38))),
            child: Text(text.isEmpty ? 'Tryck start och tala naturligt.' : text,
                style:
                    Theme.of(c).textTheme.titleLarge?.copyWith(height: 1.5))),
        const SizedBox(height: 20),
        Row(children: [
          Expanded(
              child: FilledButton.icon(
                  onPressed: loading || (model == null && !running)
                      ? null
                      : (running ? stop : start),
                  icon: Icon(running ? Icons.stop : Icons.mic),
                  label: Text(loading
                      ? 'LADDAR…'
                      : running
                          ? 'STOPPA'
                          : 'STARTA TEST'))),
          const SizedBox(width: 12),
          IconButton(
              onPressed: () => setState(() => text = ''),
              icon: const Icon(Icons.delete_outline))
        ]),
        const SizedBox(height: 26),
        Wrap(spacing: 12, runSpacing: 12, children: [
          _metric(
              'Första text', firstMs == null ? '—' : '${firstMs!.round()} ms'),
          _metric(
              'Slutresultat', finalMs == null ? '—' : '${finalMs!.round()} ms'),
          _metric('RTF', rtf == null ? '—' : rtf!.toStringAsFixed(2))
        ])
      ]));
  Widget _metric(String a, String b) => Container(
      width: 150,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
          color: const Color(0xff151b1d),
          borderRadius: BorderRadius.circular(12)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(a, style: TextStyle(color: Colors.grey[500], fontSize: 12)),
        const SizedBox(height: 5),
        Text(b,
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold))
      ]));
}
