import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

void main() => runApp(const SttLab());

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
  const LabPage({super.key});
  @override
  State<LabPage> createState() => _LabPageState();
}

class _LabPageState extends State<LabPage> {
  final recorder = AudioRecorder();
  final host = TextEditingController(text: '10.0.2.2:8000');
  StreamSubscription<Uint8List>? mic;
  WebSocketChannel? socket;
  bool running = false;
  bool loading = false;
  String text = '';
  String status = 'Redo att testa';
  static const connectTimeout = Duration(seconds: 8);
  String model = 'qwen-0.6b';
  String language = 'sv';
  double? firstMs, finalMs, rtf;
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
    final p = host.text.split(':');
    final channel = WebSocketChannel.connect(Uri.parse(
        'ws://${p[0]}:${p.length > 1 ? p[1] : '8000'}/ws/transcribe'));
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
            decoration: const InputDecoration(
                labelText: 'FastAPI-server', prefixIcon: Icon(Icons.link))),
        const SizedBox(height: 16),
        Row(children: [
          Expanded(
              child: DropdownButtonFormField<String>(
                  isExpanded: true,
                  initialValue: model,
                  decoration: const InputDecoration(labelText: 'Modell'),
                  items: const [
                    DropdownMenuItem(
                        value: 'qwen-0.6b', child: Text('Qwen3-ASR 0.6B')),
                    DropdownMenuItem(
                        value: 'qwen-1.7b', child: Text('Qwen3-ASR 1.7B')),
                    DropdownMenuItem(
                        value: 'nemotron-0.6b', child: Text('Nemotron 3.5 ASR'))
                  ],
                  onChanged: running || loading
                      ? null
                      : (v) => setState(() => model = v!))),
          const SizedBox(width: 12),
          Expanded(
              child: DropdownButtonFormField<String>(
                  isExpanded: true,
                  initialValue: language,
                  decoration: const InputDecoration(labelText: 'Språk'),
                  items: const [
                    DropdownMenuItem(value: 'sv', child: Text('Svenska')),
                    DropdownMenuItem(value: 'en', child: Text('English')),
                    DropdownMenuItem(value: 'auto', child: Text('Auto'))
                  ],
                  onChanged: running || loading
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
                  onPressed: loading ? null : (running ? stop : start),
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
