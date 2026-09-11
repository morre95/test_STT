import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:just_audio/just_audio.dart';
import 'package:record/record.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

const lime = Color(0xffd6f36b);
const orange = Color(0xffff935c);
const muted = Color(0xff84908f);

Uri serverUri(TextEditingController host, String path) {
  var value = host.text.trim();
  if (!value.contains('://')) value = 'http://$value';
  final parsed = Uri.parse(value);
  return Uri.parse(
      'http://${parsed.host}:${parsed.hasPort ? parsed.port : 8000}$path');
}

Uri serverWsUri(TextEditingController host, String path) {
  final httpUri = serverUri(host, path);
  return httpUri.replace(scheme: 'ws');
}

String responseError(http.Response response) {
  try {
    return jsonDecode(utf8.decode(response.bodyBytes))['detail'].toString();
  } catch (_) {
    return 'HTTP ${response.statusCode}';
  }
}

Future<http.Response> collect(http.StreamedResponse response) async =>
    http.Response.bytes(await response.stream.toBytes(), response.statusCode,
        headers: response.headers);

class RecordingInfo {
  RecordingInfo(this.id, this.duration, this.name, this.reference);

  factory RecordingInfo.fromJson(Map<String, dynamic> value) {
    final meta = (value['metadata'] as Map?)?.cast<String, dynamic>() ?? {};
    final id = value['id'] as String;
    return RecordingInfo(
        id,
        (value['duration'] as num).toDouble(),
        meta['original_name']?.toString() ?? 'Inspelning ${id.substring(0, 8)}',
        (value['speaker_reference'] as Map?)?.cast<String, dynamic>());
  }

  final String id;
  final double duration;
  final String name;
  final Map<String, dynamic>? reference;
}

class CatalogChoice {
  const CatalogChoice(this.id, this.name, this.available, this.languages);

  factory CatalogChoice.fromJson(Map<String, dynamic> value) => CatalogChoice(
      value['id'] as String,
      value['name'] as String,
      value['available'] == true,
      ((value['languages'] as List?) ?? const []).cast<String>());

  final String id;
  final String name;
  final bool available;
  final List<String> languages;
}

class SpeakerProfile {
  SpeakerProfile(this.id, this.name, this.samples);

  factory SpeakerProfile.fromJson(Map<String, dynamic> value) => SpeakerProfile(
      value['id'] as String,
      value['name'] as String,
      ((value['samples'] as List?) ?? const [])
          .map((item) => (item as Map).cast<String, dynamic>())
          .toList());

  final String id;
  final String name;
  final List<Map<String, dynamic>> samples;
}

class ServerField extends StatelessWidget {
  const ServerField(
      {super.key, required this.controller, required this.onRefresh});

  final TextEditingController controller;
  final VoidCallback onRefresh;

  @override
  Widget build(BuildContext context) => TextField(
      controller: controller,
      onSubmitted: (_) => onRefresh(),
      decoration: InputDecoration(
          labelText: 'FastAPI-server',
          prefixIcon: const Icon(Icons.link),
          suffixIcon: IconButton(
              onPressed: onRefresh,
              tooltip: 'Uppdatera',
              icon: const Icon(Icons.sync))));
}

class SpeakerBenchmarkPage extends StatefulWidget {
  const SpeakerBenchmarkPage({super.key, this.client});

  final http.Client? client;

  @override
  State<SpeakerBenchmarkPage> createState() => _SpeakerBenchmarkPageState();
}

class _SpeakerBenchmarkPageState extends State<SpeakerBenchmarkPage> {
  late final http.Client client = widget.client ?? http.Client();
  final host = TextEditingController(text: '10.0.2.2:8000');
  final player = AudioPlayer();
  final liveRecorder = AudioRecorder();
  StreamSubscription<Uint8List>? liveMicrophone;
  StreamSubscription? liveEvents;
  WebSocketChannel? liveSocket;
  List<RecordingInfo> recordings = [];
  List<CatalogChoice> embeddings = [];
  List<CatalogChoice> sttModels = [];
  List<SpeakerProfile> profiles = [];
  final selectedModels = <String>{};
  final thresholds = <String, double>{};
  List<Map<String, dynamic>> speakers = [];
  List<Map<String, dynamic>> segments = [];
  String? recordingId;
  String? sttModel;
  String language = 'sv';
  String status = 'Synkroniserar…';
  Map<String, dynamic>? job;
  Timer? timer;
  bool loading = false;
  String? liveModel;
  bool liveLoading = false;
  bool liveRunning = false;
  String liveStatus = 'Registrera minst ett röstprov i Profiler';
  String liveSpeaker = '—';
  double? liveScore;
  double? liveThreshold;
  double? liveProcessingMs;
  List<Map<String, dynamic>> liveScores = [];

  RecordingInfo? get selectedRecording =>
      recordings.where((item) => item.id == recordingId).firstOrNull;

  CatalogChoice? get selectedStt =>
      sttModels.where((item) => item.id == sttModel).firstOrNull;

  @override
  void initState() {
    super.initState();
    load();
  }

  Future<void> load() async {
    setState(() => loading = true);
    try {
      final values = await Future.wait([
        client.get(serverUri(host, '/recordings')),
        client.get(serverUri(host, '/speaker-models')),
        client.get(serverUri(host, '/models')),
        client.get(serverUri(host, '/speaker-profiles')),
      ]);
      final failed = values.where((item) => item.statusCode != 200).firstOrNull;
      if (failed != null) throw Exception(responseError(failed));
      final nextRecordings =
          (jsonDecode(utf8.decode(values[0].bodyBytes)) as List)
              .map((item) =>
                  RecordingInfo.fromJson((item as Map).cast<String, dynamic>()))
              .where((item) => item.duration > 0)
              .toList();
      final nextEmbeddings =
          (jsonDecode(utf8.decode(values[1].bodyBytes)) as List)
              .map((item) =>
                  CatalogChoice.fromJson((item as Map).cast<String, dynamic>()))
              .toList();
      final nextStt = (jsonDecode(utf8.decode(values[2].bodyBytes)) as List)
          .map((item) =>
              CatalogChoice.fromJson((item as Map).cast<String, dynamic>()))
          .toList();
      final nextProfiles = (jsonDecode(utf8.decode(values[3].bodyBytes))
              as List)
          .map((item) =>
              SpeakerProfile.fromJson((item as Map).cast<String, dynamic>()))
          .toList();
      if (!mounted) return;
      setState(() {
        recordings = nextRecordings;
        embeddings = nextEmbeddings;
        sttModels = nextStt;
        profiles = nextProfiles;
        selectedModels.removeWhere((id) =>
            !nextEmbeddings.any((item) => item.id == id && item.available));
        if (selectedModels.isEmpty) {
          selectedModels.addAll(nextEmbeddings
              .where((item) => item.available)
              .map((item) => item.id));
        }
        if (!nextEmbeddings
            .any((item) => item.id == liveModel && item.available)) {
          liveModel =
              nextEmbeddings.where((item) => item.available).firstOrNull?.id;
        }
        if (!nextStt.any((item) => item.id == sttModel && item.available)) {
          sttModel = nextStt.where((item) => item.available).firstOrNull?.id;
        }
        if (!nextRecordings.any((item) => item.id == recordingId)) {
          recordingId = nextRecordings.firstOrNull?.id;
        }
        if (selectedStt != null && !selectedStt!.languages.contains(language)) {
          language = selectedStt!.languages.first;
        }
        readReference();
        loading = false;
        status = nextRecordings.isEmpty
            ? 'Ladda upp ett möte för att börja'
            : 'Redo för talarbenchmark';
      });
      await loadAudio();
    } catch (error) {
      if (mounted) {
        setState(() {
          loading = false;
          status = 'Kunde inte hämta data: $error';
        });
      }
    }
  }

  void readReference() {
    final value = selectedRecording?.reference;
    speakers = ((value?['speakers'] as List?) ?? const [])
        .map((item) => Map<String, dynamic>.from(item as Map))
        .toList();
    segments = ((value?['segments'] as List?) ?? const [])
        .map((item) => Map<String, dynamic>.from(item as Map))
        .toList();
  }

  Future<void> selectRecording(String? value) async {
    setState(() {
      recordingId = value;
      readReference();
      job = null;
    });
    await loadAudio();
  }

  Future<void> loadAudio() async {
    if (recordingId == null) return;
    try {
      await player
          .setUrl(serverUri(host, '/recordings/$recordingId/audio').toString());
    } catch (error) {
      if (mounted) setState(() => status = 'Ljudfel: $error');
    }
  }

  Future<void> uploadRecording() async {
    final picked = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const ['wav', 'flac'],
        withReadStream: true);
    if (picked == null || picked.files.single.readStream == null) return;
    final file = picked.files.single;
    setState(() => status = 'Laddar upp ${file.name}…');
    final request =
        http.MultipartRequest('POST', serverUri(host, '/recordings/upload'));
    request.files.add(http.MultipartFile(
        'file', http.ByteStream(file.readStream!), file.size,
        filename: file.name));
    final response = await collect(await client.send(request));
    if (response.statusCode != 201) {
      setState(() => status = responseError(response));
      return;
    }
    final created = RecordingInfo.fromJson(
        (jsonDecode(response.body) as Map).cast<String, dynamic>());
    await load();
    await selectRecording(created.id);
  }

  Future<void> addSpeaker() async {
    var name = '';
    String? profileId;
    final value = await showDialog<Map<String, dynamic>>(
        context: context,
        builder: (dialogContext) => StatefulBuilder(
            builder: (context, update) => AlertDialog(
                    title: const Text('Ny facittalare'),
                    content: Column(mainAxisSize: MainAxisSize.min, children: [
                      TextField(
                          autofocus: true,
                          onChanged: (value) => name = value,
                          decoration: const InputDecoration(labelText: 'Namn')),
                      const SizedBox(height: 12),
                      DropdownButtonFormField<String?>(
                          initialValue: profileId,
                          decoration:
                              const InputDecoration(labelText: 'Känd profil'),
                          items: [
                            const DropdownMenuItem(
                                value: null,
                                child: Text('Okänd / ingen profil')),
                            for (final profile in profiles)
                              DropdownMenuItem(
                                  value: profile.id, child: Text(profile.name))
                          ],
                          onChanged: (next) => update(() => profileId = next))
                    ]),
                    actions: [
                      TextButton(
                          onPressed: () => Navigator.pop(dialogContext),
                          child: const Text('AVBRYT')),
                      FilledButton(
                          onPressed: () {
                            if (name.trim().isEmpty) return;
                            Navigator.pop(dialogContext, {
                              'id':
                                  'speaker-${DateTime.now().microsecondsSinceEpoch}',
                              'label': name.trim(),
                              'profile_id': profileId
                            });
                          },
                          child: const Text('LÄGG TILL'))
                    ])));
    if (value != null) setState(() => speakers.add(value));
  }

  Future<void> addSegment() async {
    if (speakers.isEmpty) {
      setState(() => status = 'Lägg först till en facittalare');
      return;
    }
    var begin = (player.position.inMilliseconds / 1000).toStringAsFixed(2);
    var finish = ((player.position.inMilliseconds + 2000)
                .clamp(0, (selectedRecording?.duration ?? 2) * 1000) /
            1000)
        .toStringAsFixed(2);
    var words = '';
    String speakerId = speakers.first['id'] as String;
    final value = await showDialog<Map<String, dynamic>>(
        context: context,
        builder: (dialogContext) => StatefulBuilder(
            builder: (context, update) => AlertDialog(
                    title: const Text('Nytt facitsegment'),
                    content: SingleChildScrollView(
                        child:
                            Column(mainAxisSize: MainAxisSize.min, children: [
                      DropdownButtonFormField<String>(
                          initialValue: speakerId,
                          decoration:
                              const InputDecoration(labelText: 'Talare'),
                          items: [
                            for (final speaker in speakers)
                              DropdownMenuItem(
                                  value: speaker['id'] as String,
                                  child: Text(speaker['label'] as String))
                          ],
                          onChanged: (next) => update(() => speakerId = next!)),
                      const SizedBox(height: 12),
                      Row(children: [
                        Expanded(
                            child: TextFormField(
                                initialValue: begin,
                                onChanged: (value) => begin = value,
                                keyboardType: TextInputType.number,
                                decoration: const InputDecoration(
                                    labelText: 'Start (s)'))),
                        const SizedBox(width: 8),
                        Expanded(
                            child: TextFormField(
                                initialValue: finish,
                                onChanged: (value) => finish = value,
                                keyboardType: TextInputType.number,
                                decoration: const InputDecoration(
                                    labelText: 'Slut (s)')))
                      ]),
                      const SizedBox(height: 12),
                      TextFormField(
                          onChanged: (value) => words = value,
                          minLines: 2,
                          maxLines: 5,
                          decoration:
                              const InputDecoration(labelText: 'Det som sägs'))
                    ])),
                    actions: [
                      TextButton(
                          onPressed: () => Navigator.pop(dialogContext),
                          child: const Text('AVBRYT')),
                      FilledButton(
                          onPressed: () {
                            final start =
                                double.tryParse(begin.replaceAll(',', '.'));
                            final end =
                                double.tryParse(finish.replaceAll(',', '.'));
                            if (start == null ||
                                end == null ||
                                start < 0 ||
                                end <= start ||
                                words.trim().isEmpty) {
                              return;
                            }
                            Navigator.pop(dialogContext, {
                              'start_ms': (start * 1000).round(),
                              'end_ms': (end * 1000).round(),
                              'speaker_id': speakerId,
                              'text': words.trim()
                            });
                          },
                          child: const Text('LÄGG TILL'))
                    ])));
    if (value != null) {
      setState(() {
        segments.add(value);
        segments.sort((left, right) =>
            (left['start_ms'] as int).compareTo(right['start_ms'] as int));
      });
    }
  }

  Future<bool> saveReference() async {
    if (recordingId == null || speakers.isEmpty || segments.isEmpty) {
      return true;
    }
    final response = await client.put(
        serverUri(host, '/recordings/$recordingId/speaker-reference'),
        headers: {'content-type': 'application/json'},
        body: jsonEncode({'speakers': speakers, 'segments': segments}));
    if (response.statusCode != 200) {
      setState(() => status = responseError(response));
      return false;
    }
    setState(() => status = 'Facit sparat');
    return true;
  }

  Future<void> startBenchmark() async {
    if (recordingId == null || sttModel == null || selectedModels.isEmpty) {
      return;
    }
    if (!await saveReference()) return;
    final response = await client.post(serverUri(host, '/speaker-benchmarks'),
        headers: {'content-type': 'application/json'},
        body: jsonEncode({
          'recording_id': recordingId,
          'embedding_models': selectedModels.toList(),
          'stt_model': sttModel,
          'language': language,
          'thresholds': thresholds
        }));
    if (response.statusCode != 202) {
      setState(() => status = responseError(response));
      return;
    }
    setState(
        () => job = (jsonDecode(response.body) as Map).cast<String, dynamic>());
    timer?.cancel();
    timer = Timer.periodic(const Duration(seconds: 1), (_) => poll());
    await poll();
  }

  Future<void> poll() async {
    final id = job?['id'];
    if (id == null) return;
    final response =
        await client.get(serverUri(host, '/speaker-benchmarks/$id'));
    if (response.statusCode != 200 || !mounted) return;
    final value = (jsonDecode(response.body) as Map).cast<String, dynamic>();
    setState(() {
      job = value;
      status = value['stage']?.toString() ?? value['status'].toString();
    });
    if (!{'queued', 'running'}.contains(value['status'])) timer?.cancel();
  }

  Future<void> cancel() async {
    await client
        .post(serverUri(host, '/speaker-benchmarks/${job!['id']}/cancel'));
    await poll();
  }

  Future<void> _beginLiveMicrophone() async {
    final stream = await liveRecorder.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits, sampleRate: 16000, numChannels: 1));
    liveMicrophone = stream.listen((bytes) {
      for (var offset = 0; offset < bytes.length; offset += 3200) {
        final end = offset + 3200 < bytes.length ? offset + 3200 : bytes.length;
        liveSocket?.sink.add(Uint8List.sublistView(bytes, offset, end));
      }
    });
    if (!mounted) return;
    setState(() {
      liveLoading = false;
      liveRunning = true;
      liveStatus = 'Lyssnar · nytt resultat var 0,75 s';
    });
  }

  Future<void> startLive() async {
    if (liveModel == null) return;
    if (!profiles.any((profile) => profile.samples.isNotEmpty)) {
      setState(() => liveStatus = 'Lägg till minst ett röstprov i Profiler');
      return;
    }
    if (!await liveRecorder.hasPermission()) {
      setState(() => liveStatus = 'Mikrofonbehörighet saknas');
      return;
    }
    final channel =
        WebSocketChannel.connect(serverWsUri(host, '/ws/speaker-identify'));
    liveSocket = channel;
    setState(() {
      liveLoading = true;
      liveSpeaker = '—';
      liveScore = liveThreshold = liveProcessingMs = null;
      liveScores = [];
      liveStatus = 'Ansluter…';
    });
    liveEvents = channel.stream.listen((raw) async {
      final message =
          (jsonDecode(raw as String) as Map).cast<String, dynamic>();
      if (!mounted) return;
      if (message['type'] == 'loading') {
        setState(() => liveStatus = 'Laddar modell och röstprofiler…');
      } else if (message['type'] == 'ready') {
        setState(() {
          liveThreshold = (message['threshold'] as num?)?.toDouble();
          liveStatus = 'Startar mikrofon…';
        });
        await _beginLiveMicrophone();
      } else if (message['type'] == 'speaker') {
        final speech = message['speech'] == true;
        setState(() {
          liveSpeaker =
              speech ? (message['speaker']?.toString() ?? 'Okänd') : '—';
          liveScore = (message['score'] as num?)?.toDouble();
          liveProcessingMs = (message['processing_ms'] as num?)?.toDouble();
          liveScores = ((message['scores'] as List?) ?? const [])
              .map((item) => (item as Map).cast<String, dynamic>())
              .toList();
          liveStatus = speech ? 'Röst identifierad' : 'Lyssnar · inget tal';
        });
      } else if (message['type'] == 'error') {
        await _stopLiveCapture();
        if (mounted) {
          setState(
              () => liveStatus = message['message']?.toString() ?? 'Modellfel');
        }
      } else if (message['type'] == 'stopped') {
        await _stopLiveCapture();
      }
    }, onError: (error) async {
      await _stopLiveCapture();
      if (mounted) setState(() => liveStatus = 'Anslutningen bröts: $error');
    }, onDone: () async {
      await _stopLiveCapture();
    });
    try {
      await channel.ready.timeout(const Duration(seconds: 8));
      channel.sink.add(jsonEncode({'type': 'start', 'model': liveModel}));
    } catch (error) {
      await _stopLiveCapture();
      if (mounted) setState(() => liveStatus = 'Kunde inte ansluta: $error');
    }
  }

  Future<void> _stopLiveCapture() async {
    await liveMicrophone?.cancel();
    liveMicrophone = null;
    if (liveRunning) await liveRecorder.stop();
    if (mounted) {
      setState(() {
        liveLoading = false;
        liveRunning = false;
      });
    }
  }

  Future<void> stopLive() async {
    await _stopLiveCapture();
    liveSocket?.sink.add(jsonEncode({'type': 'stop'}));
    if (mounted) setState(() => liveStatus = 'Stoppad');
  }

  @override
  void dispose() {
    timer?.cancel();
    liveMicrophone?.cancel();
    liveEvents?.cancel();
    liveSocket?.sink.close();
    liveRecorder.dispose();
    player.dispose();
    host.dispose();
    client.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final active = {'queued', 'running'}.contains(job?['status']);
    return Material(
        color: Colors.transparent,
        child: ListView(padding: const EdgeInsets.all(24), children: [
          const Text('Vem sade vad?',
              style: TextStyle(fontSize: 29, fontWeight: FontWeight.w900)),
          const SizedBox(height: 7),
          const Text(
              'Samma VAD och klustring. Fyra röstavtryck. Ett ärligt facit.',
              style: TextStyle(color: muted)),
          const SizedBox(height: 22),
          ServerField(controller: host, onRefresh: load),
          const SizedBox(height: 14),
          LabSection(
              number: 'LIVE',
              title: 'IDENTIFIERING',
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const Text(
                        'Välj en modell och identifiera registrerade profiler direkt från mikrofonen.',
                        style: TextStyle(color: muted, fontSize: 12)),
                    const SizedBox(height: 12),
                    DropdownButtonFormField<String>(
                        isExpanded: true,
                        initialValue: liveModel,
                        decoration:
                            const InputDecoration(labelText: 'Live-modell'),
                        items: [
                          for (final model in embeddings)
                            DropdownMenuItem(
                                value: model.id,
                                enabled: model.available,
                                child: Text(model.available
                                    ? model.name
                                    : '${model.name} (saknas)'))
                        ],
                        onChanged: liveLoading || liveRunning
                            ? null
                            : (value) => setState(() => liveModel = value)),
                    const SizedBox(height: 14),
                    Text(liveSpeaker,
                        textAlign: TextAlign.center,
                        style: TextStyle(
                            color: liveSpeaker == 'Okänd' ? orange : lime,
                            fontSize: 31,
                            fontWeight: FontWeight.w900)),
                    if (liveScore != null)
                      Text(
                          'likhet ${liveScore!.toStringAsFixed(3)} · gräns ${liveThreshold?.toStringAsFixed(3) ?? '—'} · ${liveProcessingMs?.toStringAsFixed(0) ?? '—'} ms',
                          textAlign: TextAlign.center,
                          style: const TextStyle(color: muted, fontSize: 11)),
                    if (liveScores.isNotEmpty) ...[
                      const SizedBox(height: 9),
                      Wrap(
                          alignment: WrapAlignment.center,
                          spacing: 6,
                          runSpacing: 6,
                          children: [
                            for (final score in liveScores)
                              MetricTag(
                                  label: score['speaker'].toString(),
                                  value: (score['score'] as num)
                                      .toStringAsFixed(3))
                          ])
                    ],
                    const SizedBox(height: 12),
                    FilledButton.icon(
                        onPressed: loading || liveModel == null
                            ? null
                            : liveRunning || liveLoading
                                ? stopLive
                                : startLive,
                        icon: Icon(liveRunning || liveLoading
                            ? Icons.stop_circle_outlined
                            : Icons.mic),
                        label: Text(liveRunning || liveLoading
                            ? 'STOPPA LIVE'
                            : 'STARTA LIVE')),
                    const SizedBox(height: 7),
                    Text(liveStatus,
                        textAlign: TextAlign.center,
                        style: const TextStyle(color: muted, fontSize: 11))
                  ])),
          const SizedBox(height: 14),
          Row(children: [
            Expanded(
                child: DropdownButtonFormField<String>(
                    isExpanded: true,
                    initialValue: recordingId,
                    decoration: const InputDecoration(labelText: 'Mötesljud'),
                    items: [
                      for (final item in recordings)
                        DropdownMenuItem(value: item.id, child: Text(item.name))
                    ],
                    onChanged: active ? null : selectRecording)),
            const SizedBox(width: 9),
            IconButton.filledTonal(
                onPressed: active ? null : uploadRecording,
                tooltip: 'Ladda upp WAV/FLAC',
                icon: const Icon(Icons.upload_file))
          ]),
          if (selectedRecording != null) ...[
            const SizedBox(height: 10),
            AudioStrip(player: player, duration: selectedRecording!.duration)
          ],
          const SizedBox(height: 14),
          LabSection(
              number: '01',
              title: 'FACIT',
              trailing: Wrap(spacing: 3, children: [
                TextButton.icon(
                    onPressed: active ? null : addSpeaker,
                    icon: const Icon(Icons.person_add_alt_1, size: 17),
                    label: const Text('TALARE')),
                TextButton.icon(
                    onPressed: active ? null : addSegment,
                    icon: const Icon(Icons.add, size: 17),
                    label: const Text('SEGMENT')),
                IconButton(
                    onPressed: active ? null : saveReference,
                    icon: const Icon(Icons.save_outlined))
              ]),
              child: speakers.isEmpty
                  ? const EmptyState(
                      text: 'Lägg till talare och deras tidsatta repliker.')
                  : Column(children: [
                      Wrap(spacing: 7, runSpacing: 7, children: [
                        for (final speaker in speakers)
                          InputChip(
                              avatar: Icon(
                                  speaker['profile_id'] == null
                                      ? Icons.person_outline
                                      : Icons.verified_user_outlined,
                                  size: 16),
                              label: Text(speaker['label'] as String),
                              onDeleted: active
                                  ? null
                                  : () => setState(() {
                                        segments.removeWhere((item) =>
                                            item['speaker_id'] ==
                                            speaker['id']);
                                        speakers.remove(speaker);
                                      }))
                      ]),
                      const SizedBox(height: 8),
                      for (final segment in segments)
                        ReferenceRow(
                            segment: segment,
                            speaker: speakers.firstWhere(
                                (item) => item['id'] == segment['speaker_id']),
                            onDelete: active
                                ? null
                                : () =>
                                    setState(() => segments.remove(segment)))
                    ])),
          const SizedBox(height: 14),
          LabSection(
              number: '02',
              title: 'KÖRNING',
              child: Column(children: [
                for (final model in embeddings)
                  CheckboxListTile(
                      dense: true,
                      contentPadding: EdgeInsets.zero,
                      value: selectedModels.contains(model.id),
                      onChanged: !model.available || active
                          ? null
                          : (checked) => setState(() => checked == true
                              ? selectedModels.add(model.id)
                              : selectedModels.remove(model.id)),
                      title: Text(model.name),
                      subtitle: model.available
                          ? null
                          : const Text('Runtime saknas',
                              style: TextStyle(color: orange))),
                const Divider(),
                Row(children: [
                  Expanded(
                      child: DropdownButtonFormField<String>(
                          isExpanded: true,
                          initialValue: sttModel,
                          decoration: const InputDecoration(
                              labelText: 'Fast STT-modell'),
                          items: [
                            for (final model in sttModels)
                              DropdownMenuItem(
                                  value: model.id,
                                  enabled: model.available,
                                  child: Text(model.available
                                      ? model.name
                                      : '${model.name} (saknas)'))
                          ],
                          onChanged: active
                              ? null
                              : (value) => setState(() {
                                    sttModel = value;
                                    if (selectedStt != null &&
                                        !selectedStt!.languages
                                            .contains(language)) {
                                      language = selectedStt!.languages.first;
                                    }
                                  }))),
                  const SizedBox(width: 9),
                  SizedBox(
                      width: 118,
                      child: DropdownButtonFormField<String>(
                          initialValue: selectedStt == null ? null : language,
                          decoration: const InputDecoration(labelText: 'Språk'),
                          items: [
                            for (final code
                                in selectedStt?.languages ?? const <String>[])
                              DropdownMenuItem(
                                  value: code, child: Text(code.toUpperCase()))
                          ],
                          onChanged: active
                              ? null
                              : (value) => setState(() => language = value!)))
                ]),
                ExpansionTile(
                    tilePadding: EdgeInsets.zero,
                    title: const Text('Manuella identitetströsklar',
                        style: TextStyle(fontSize: 12)),
                    children: [
                      for (final model in embeddings
                          .where((item) => selectedModels.contains(item.id)))
                        Padding(
                            padding: const EdgeInsets.only(bottom: 8),
                            child: TextFormField(
                                initialValue: thresholds[model.id]?.toString(),
                                decoration: InputDecoration(
                                    labelText: '${model.name} (−1…1)'),
                                keyboardType: TextInputType.number,
                                onChanged: (value) {
                                  final parsed = double.tryParse(
                                      value.replaceAll(',', '.'));
                                  if (parsed == null) {
                                    thresholds.remove(model.id);
                                  } else {
                                    thresholds[model.id] = parsed.clamp(-1, 1);
                                  }
                                }))
                    ]),
                if (active) ...[
                  const SizedBox(height: 8),
                  LinearProgressIndicator(
                      value: (job?['progress'] as num?)?.toDouble(),
                      color: orange),
                  const SizedBox(height: 10)
                ],
                Row(children: [
                  Expanded(
                      child: FilledButton.icon(
                          onPressed: active ||
                                  loading ||
                                  recordingId == null ||
                                  sttModel == null ||
                                  selectedModels.isEmpty
                              ? null
                              : startBenchmark,
                          icon: const Icon(Icons.play_arrow),
                          label: Text(active
                              ? status.toUpperCase()
                              : 'KÖR ALLA VALDA'))),
                  if (active) ...[
                    const SizedBox(width: 8),
                    IconButton.outlined(
                        onPressed: cancel, icon: const Icon(Icons.stop))
                  ]
                ])
              ])),
          if ((job?['results'] as List?)?.isNotEmpty == true) ...[
            const SizedBox(height: 14),
            LabSection(
                number: '03',
                title: 'RESULTAT',
                child: Column(children: [
                  if (job?['oracle'] != null)
                    MetricTag(
                        label: 'ORACLE WER',
                        value: percent(job!['oracle']['wer'])),
                  for (final raw in job!['results'] as List)
                    ResultCard(
                        result: (raw as Map).cast<String, dynamic>(),
                        duration: selectedRecording?.duration ?? 1)
                ]))
          ],
          const SizedBox(height: 18),
          Text(status,
              textAlign: TextAlign.center,
              style: TextStyle(color: active ? orange : muted, fontSize: 12))
        ]));
  }
}

class SpeakerProfilesPage extends StatefulWidget {
  const SpeakerProfilesPage({super.key, this.client});

  final http.Client? client;

  @override
  State<SpeakerProfilesPage> createState() => _SpeakerProfilesPageState();
}

class _SpeakerProfilesPageState extends State<SpeakerProfilesPage> {
  late final http.Client client = widget.client ?? http.Client();
  final host = TextEditingController(text: '10.0.2.2:8000');
  final recorder = AudioRecorder();
  final captured = <int>[];
  StreamSubscription<Uint8List>? microphone;
  List<SpeakerProfile> profiles = [];
  String? activeProfile;
  String status = 'Två profiler med två prov var krävs för auto-kalibrering';

  @override
  void initState() {
    super.initState();
    load();
  }

  Future<void> load() async {
    try {
      final response = await client.get(serverUri(host, '/speaker-profiles'));
      if (response.statusCode != 200) throw Exception(responseError(response));
      if (!mounted) return;
      setState(() {
        profiles = (jsonDecode(utf8.decode(response.bodyBytes)) as List)
            .map((item) =>
                SpeakerProfile.fromJson((item as Map).cast<String, dynamic>()))
            .toList();
        status = profiles.isEmpty
            ? 'Skapa din första talarprofil'
            : 'Profiler synkroniserade';
      });
    } catch (error) {
      if (mounted) setState(() => status = 'Kunde inte hämta profiler: $error');
    }
  }

  Future<void> createProfile() async {
    var value = '';
    final name = await showDialog<String>(
        context: context,
        builder: (dialogContext) => AlertDialog(
                title: const Text('Ny talarprofil'),
                content: TextField(
                    autofocus: true,
                    onChanged: (next) => value = next,
                    decoration: const InputDecoration(labelText: 'Namn')),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(dialogContext),
                      child: const Text('AVBRYT')),
                  FilledButton(
                      onPressed: () =>
                          Navigator.pop(dialogContext, value.trim()),
                      child: const Text('SKAPA'))
                ]));
    if (name == null || name.isEmpty) return;
    final response = await client.post(serverUri(host, '/speaker-profiles'),
        headers: {'content-type': 'application/json'},
        body: jsonEncode({'name': name}));
    setState(() => status =
        response.statusCode == 201 ? 'Profil skapad' : responseError(response));
    await load();
  }

  Future<void> uploadSample(SpeakerProfile profile) async {
    final picked = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const ['wav', 'flac'],
        withReadStream: true);
    if (picked == null || picked.files.single.readStream == null) return;
    final file = picked.files.single;
    final request = http.MultipartRequest(
        'POST', serverUri(host, '/speaker-profiles/${profile.id}/samples'));
    request.files.add(http.MultipartFile(
        'file', http.ByteStream(file.readStream!), file.size,
        filename: file.name));
    setState(() => status = 'Laddar upp röstprov…');
    final response = await collect(await client.send(request));
    setState(() => status = response.statusCode == 201
        ? 'Röstprov sparat'
        : responseError(response));
    await load();
  }

  Future<void> toggleRecording(SpeakerProfile profile) async {
    if (activeProfile == profile.id) {
      await microphone?.cancel();
      await recorder.stop();
      setState(() => activeProfile = null);
      if (captured.length < 64000) {
        setState(() => status = 'Spela in minst två sekunder');
        return;
      }
      final request = http.MultipartRequest(
          'POST', serverUri(host, '/speaker-profiles/${profile.id}/samples'));
      request.files.add(http.MultipartFile.fromBytes(
          'file', makeWav(Uint8List.fromList(captured)),
          filename: 'profil-${DateTime.now().millisecondsSinceEpoch}.wav'));
      final response = await collect(await client.send(request));
      setState(() => status = response.statusCode == 201
          ? 'Röstprov sparat'
          : responseError(response));
      await load();
      return;
    }
    if (!await recorder.hasPermission()) {
      setState(() => status = 'Mikrofonbehörighet saknas');
      return;
    }
    captured.clear();
    final stream = await recorder.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits, sampleRate: 16000, numChannels: 1));
    microphone = stream.listen(captured.addAll);
    setState(() {
      activeProfile = profile.id;
      status = 'Spelar in ${profile.name}…';
    });
  }

  Uint8List makeWav(Uint8List pcm) {
    final data = ByteData(44 + pcm.length);
    void putText(int offset, String value) {
      for (var index = 0; index < value.length; index++) {
        data.setUint8(offset + index, value.codeUnitAt(index));
      }
    }

    putText(0, 'RIFF');
    data.setUint32(4, 36 + pcm.length, Endian.little);
    putText(8, 'WAVEfmt ');
    data.setUint32(16, 16, Endian.little);
    data.setUint16(20, 1, Endian.little);
    data.setUint16(22, 1, Endian.little);
    data.setUint32(24, 16000, Endian.little);
    data.setUint32(28, 32000, Endian.little);
    data.setUint16(32, 2, Endian.little);
    data.setUint16(34, 16, Endian.little);
    putText(36, 'data');
    data.setUint32(40, pcm.length, Endian.little);
    data.buffer.asUint8List(44).setAll(0, pcm);
    return data.buffer.asUint8List();
  }

  Future<void> deleteProfile(SpeakerProfile profile) async {
    final response =
        await client.delete(serverUri(host, '/speaker-profiles/${profile.id}'));
    setState(() => status = response.statusCode == 200
        ? 'Profil borttagen'
        : responseError(response));
    await load();
  }

  Future<void> deleteSample(SpeakerProfile profile, String sampleId) async {
    final response = await client.delete(
        serverUri(host, '/speaker-profiles/${profile.id}/samples/$sampleId'));
    setState(() => status = response.statusCode == 200
        ? 'Röstprov borttaget'
        : responseError(response));
    await load();
  }

  @override
  void dispose() {
    microphone?.cancel();
    recorder.dispose();
    host.dispose();
    client.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Material(
      color: Colors.transparent,
      child: ListView(padding: const EdgeInsets.all(24), children: [
        const Text('Röstarkiv',
            style: TextStyle(fontSize: 29, fontWeight: FontWeight.w900)),
        const SizedBox(height: 7),
        const Text('Separata referensklipp håller benchmarken ärlig.',
            style: TextStyle(color: muted)),
        const SizedBox(height: 22),
        ServerField(controller: host, onRefresh: load),
        const SizedBox(height: 14),
        Row(children: [
          Expanded(child: Text(status, style: const TextStyle(color: muted))),
          FilledButton.icon(
              onPressed: activeProfile == null ? createProfile : null,
              icon: const Icon(Icons.person_add_alt),
              label: const Text('NY PROFIL'))
        ]),
        const SizedBox(height: 14),
        if (profiles.isEmpty)
          const EmptyState(text: 'Inga röster registrerade ännu.')
        else
          for (final profile in profiles)
            Padding(
                padding: const EdgeInsets.only(bottom: 11),
                child: Card(
                    child: Padding(
                        padding: const EdgeInsets.all(15),
                        child: Column(children: [
                          Row(children: [
                            CircleAvatar(
                                backgroundColor: profile.samples.length >= 2
                                    ? lime
                                    : const Color(0xff2c3538),
                                foregroundColor: Colors.black,
                                child: Text(profile.name.characters.first
                                    .toUpperCase())),
                            const SizedBox(width: 11),
                            Expanded(
                                child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                  Text(profile.name,
                                      style: const TextStyle(
                                          fontSize: 17,
                                          fontWeight: FontWeight.w900)),
                                  Text(
                                      '${profile.samples.length}/2 prov för kalibrering',
                                      style: TextStyle(
                                          color: profile.samples.length >= 2
                                              ? lime
                                              : orange,
                                          fontSize: 11))
                                ])),
                            IconButton(
                                onPressed: activeProfile == null
                                    ? () => uploadSample(profile)
                                    : null,
                                tooltip: 'Ladda upp',
                                icon: const Icon(Icons.upload_file)),
                            IconButton(
                                onPressed: activeProfile == null ||
                                        activeProfile == profile.id
                                    ? () => toggleRecording(profile)
                                    : null,
                                color:
                                    activeProfile == profile.id ? orange : null,
                                tooltip: activeProfile == profile.id
                                    ? 'Stoppa'
                                    : 'Spela in',
                                icon: Icon(activeProfile == profile.id
                                    ? Icons.stop_circle_outlined
                                    : Icons.mic_none)),
                            IconButton(
                                onPressed: activeProfile == null
                                    ? () => deleteProfile(profile)
                                    : null,
                                icon: const Icon(Icons.delete_outline))
                          ]),
                          if (profile.samples.isNotEmpty) ...[
                            const Divider(height: 20),
                            for (var index = 0;
                                index < profile.samples.length;
                                index++)
                              Row(children: [
                                const Icon(Icons.multitrack_audio,
                                    size: 16, color: muted),
                                const SizedBox(width: 7),
                                Expanded(
                                    child: Text(
                                        'Prov ${index + 1} · ${(profile.samples[index]['duration'] as num).toStringAsFixed(1)} s',
                                        style: const TextStyle(fontSize: 11))),
                                IconButton(
                                    visualDensity: VisualDensity.compact,
                                    onPressed: () => deleteSample(profile,
                                        profile.samples[index]['id'] as String),
                                    icon: const Icon(Icons.close, size: 17))
                              ])
                          ]
                        ]))))
      ]));
}

class AudioStrip extends StatelessWidget {
  const AudioStrip({super.key, required this.player, required this.duration});

  final AudioPlayer player;
  final double duration;

  @override
  Widget build(BuildContext context) => Card(
      child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          child: Row(children: [
            StreamBuilder<PlayerState>(
                stream: player.playerStateStream,
                builder: (context, snapshot) => IconButton(
                    onPressed: snapshot.data?.playing == true
                        ? player.pause
                        : player.play,
                    icon: Icon(snapshot.data?.playing == true
                        ? Icons.pause
                        : Icons.play_arrow))),
            Expanded(
                child: StreamBuilder<Duration>(
                    stream: player.positionStream,
                    builder: (context, snapshot) {
                      final position =
                          snapshot.data?.inMilliseconds.toDouble() ?? 0;
                      return Slider(
                          value: position.clamp(0, duration * 1000),
                          max: (duration * 1000).clamp(1, double.infinity),
                          onChanged: (value) => player
                              .seek(Duration(milliseconds: value.round())));
                    })),
            StreamBuilder<Duration>(
                stream: player.positionStream,
                builder: (context, snapshot) => Text(
                    '${((snapshot.data?.inMilliseconds ?? 0) / 1000).toStringAsFixed(1)} / ${duration.toStringAsFixed(1)} s',
                    style: const TextStyle(color: muted, fontSize: 10)))
          ])));
}

class LabSection extends StatelessWidget {
  const LabSection(
      {super.key,
      required this.number,
      required this.title,
      required this.child,
      this.trailing});

  final String number;
  final String title;
  final Widget child;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) => Card(
      child: Padding(
          padding: const EdgeInsets.all(16),
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Text(number,
                  style: const TextStyle(
                      color: lime, fontWeight: FontWeight.w900, fontSize: 11)),
              const SizedBox(width: 8),
              Expanded(
                  child: Text(title,
                      style: const TextStyle(
                          fontWeight: FontWeight.w900, letterSpacing: 1.2))),
              if (trailing != null) trailing!
            ]),
            const Divider(height: 19),
            child
          ])));
}

class ReferenceRow extends StatelessWidget {
  const ReferenceRow(
      {super.key,
      required this.segment,
      required this.speaker,
      required this.onDelete});

  final Map<String, dynamic> segment;
  final Map<String, dynamic> speaker;
  final VoidCallback? onDelete;

  @override
  Widget build(BuildContext context) => Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        SizedBox(
            width: 88,
            child: Text(
                '${((segment['start_ms'] as int) / 1000).toStringAsFixed(1)}–${((segment['end_ms'] as int) / 1000).toStringAsFixed(1)}',
                style: const TextStyle(color: lime, fontSize: 10))),
        SizedBox(
            width: 82,
            child: Text(speaker['label'] as String,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                    fontWeight: FontWeight.bold, fontSize: 11))),
        Expanded(child: Text(segment['text'] as String)),
        IconButton(
            visualDensity: VisualDensity.compact,
            onPressed: onDelete,
            icon: const Icon(Icons.close, size: 16))
      ]));
}

class ResultCard extends StatelessWidget {
  const ResultCard({super.key, required this.result, required this.duration});

  final Map<String, dynamic> result;
  final double duration;

  @override
  Widget build(BuildContext context) {
    if (result['status'] == 'failed') {
      return Padding(
          padding: const EdgeInsets.only(top: 10),
          child: Text('${result['model']}: ${result['error']}',
              style: const TextStyle(color: orange)));
    }
    final metrics = (result['metrics'] as Map?)?.cast<String, dynamic>();
    final diar = (metrics?['der'] as Map?)?.cast<String, dynamic>();
    final overlap =
        (metrics?['der_with_overlap'] as Map?)?.cast<String, dynamic>();
    final identity = (metrics?['identity'] as Map?)?.cast<String, dynamic>();
    final unknown = (identity?['unknown'] as Map?)?.cast<String, dynamic>();
    final calibration =
        (result['calibration'] as Map?)?.cast<String, dynamic>();
    final timing = (result['timing'] as Map?)?.cast<String, dynamic>() ?? {};
    final parts = ((result['segments'] as List?) ?? const [])
        .map((item) => (item as Map).cast<String, dynamic>())
        .toList();
    return ExpansionTile(
        tilePadding: EdgeInsets.zero,
        initiallyExpanded: true,
        title: Text(result['model'].toString(),
            style: const TextStyle(fontWeight: FontWeight.w900)),
        subtitle: Text(
            '${result['detected_speakers']} talare · RTF ${decimal(timing['end_to_end_rtf'])}',
            style: const TextStyle(color: muted)),
        children: [
          if (calibration?['available'] == true)
            Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                        'ID-tröskel ${decimal(calibration?['threshold'])} · FAR ${percent(calibration?['far'])} · FRR ${percent(calibration?['frr'])}',
                        style: const TextStyle(color: muted, fontSize: 10))))
          else if (calibration != null)
            Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                        calibration['reason']?.toString() ??
                            'Namngivning avstängd',
                        style: const TextStyle(color: orange, fontSize: 10)))),
          Wrap(spacing: 6, runSpacing: 6, children: [
            MetricTag(label: 'DER', value: percent(diar?['der'])),
            MetricTag(label: 'DER + ÖVERLAPP', value: percent(overlap?['der'])),
            MetricTag(label: 'JER', value: percent(diar?['jer'])),
            MetricTag(label: 'cpWER', value: percent(metrics?['cpwer'])),
            MetricTag(
                label: 'SA-WER',
                value: percent(metrics?['speaker_attributed_wer'])),
            MetricTag(label: 'ID', value: percent(identity?['accuracy'])),
            MetricTag(label: 'MISS', value: percent(diar?['miss'])),
            MetricTag(
                label: 'FALSKT TAL', value: percent(diar?['false_alarm'])),
            MetricTag(label: 'FÖRVÄXLING', value: percent(diar?['confusion'])),
            MetricTag(label: 'OKÄND F1', value: percent(unknown?['f1'])),
            MetricTag(
                label: 'ANTALSFEL',
                value: metrics?['speaker_count_error']?.toString() ?? '—')
          ]),
          const SizedBox(height: 11),
          SpeakerTimeline(segments: parts, duration: duration),
          const SizedBox(height: 7),
          for (final part in parts)
            Padding(
                padding: const EdgeInsets.symmetric(vertical: 3),
                child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SizedBox(
                          width: 92,
                          child: Text(part['speaker'].toString(),
                              overflow: TextOverflow.ellipsis,
                              style:
                                  const TextStyle(color: lime, fontSize: 10))),
                      Expanded(child: Text(part['text']?.toString() ?? ''))
                    ]))
        ]);
  }
}

class SpeakerTimeline extends StatelessWidget {
  const SpeakerTimeline(
      {super.key, required this.segments, required this.duration});

  final List<Map<String, dynamic>> segments;
  final double duration;
  static const colors = [lime, orange, Color(0xff70cfff), Color(0xffff75ac)];

  @override
  Widget build(BuildContext context) => Container(
      height: 70,
      decoration: BoxDecoration(
          color: const Color(0xff0e1214),
          borderRadius: BorderRadius.circular(8)),
      child: LayoutBuilder(
          builder: (context, constraints) => Stack(children: [
                for (final part in segments)
                  Positioned(
                      left: constraints.maxWidth *
                          (part['start_ms'] as num) /
                          1000 /
                          duration,
                      width: (constraints.maxWidth *
                              ((part['end_ms'] as num) -
                                  (part['start_ms'] as num)) /
                              1000 /
                              duration)
                          .clamp(2, constraints.maxWidth),
                      top: 9 + ((part['cluster_id'] as num).toInt() % 3) * 18,
                      height: 13,
                      child: Tooltip(
                          message: '${part['speaker']} · ${part['text'] ?? ''}',
                          child: DecoratedBox(
                              decoration: BoxDecoration(
                                  color: colors[
                                      (part['cluster_id'] as num).toInt() %
                                          colors.length],
                                  borderRadius: BorderRadius.circular(3)))))
              ])));
}

class MetricTag extends StatelessWidget {
  const MetricTag({super.key, required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 7),
      decoration: BoxDecoration(
          color: const Color(0xff222a2d),
          borderRadius: BorderRadius.circular(7)),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(label, style: const TextStyle(color: muted, fontSize: 8)),
        const SizedBox(height: 2),
        Text(value, style: const TextStyle(fontWeight: FontWeight.w900))
      ]));
}

class EmptyState extends StatelessWidget {
  const EmptyState({super.key, required this.text});

  final String text;

  @override
  Widget build(BuildContext context) => Container(
      width: double.infinity,
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
          color: const Color(0xff111619),
          borderRadius: BorderRadius.circular(9)),
      child: Text(text,
          textAlign: TextAlign.center, style: const TextStyle(color: muted)));
}

String percent(dynamic value) =>
    value is num ? '${(value * 100).toStringAsFixed(1)}%' : '—';

String decimal(dynamic value) => value is num ? value.toStringAsFixed(2) : '—';
