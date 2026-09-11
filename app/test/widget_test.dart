import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/testing.dart';
import 'package:http/http.dart' as http;
import 'package:local_stt_lab/main.dart';
import 'package:local_stt_lab/speaker_lab.dart';

const catalog = [
  {
    'id': 'qwen-0.6b',
    'name': 'Qwen3-ASR 0.6B',
    'languages': ['sv', 'en', 'auto'],
    'available': true
  },
  {
    'id': 'vosk-sv',
    'name': 'Vosk Small Svenska 0.15',
    'languages': ['sv'],
    'available': true
  },
  {
    'id': 'vosk-en',
    'name': 'Vosk Small English 0.15',
    'languages': ['en'],
    'available': false
  },
];

Widget lab(List<Map<String, Object>> models) =>
    MaterialApp(home: LabPage(client: MockClient((request) async {
      expect(request.url.path, '/models');
      return http.Response(jsonEncode(models), 200,
          headers: {'content-type': 'application/json; charset=utf-8'});
    })));

void main() {
  testWidgets('visar STT-labbet', (tester) async {
    await tester.pumpWidget(const SttLab());

    expect(find.text('LOCAL STT LAB'), findsOneWidget);
    expect(find.text('Lyssna. Mät. Jämför.'), findsOneWidget);
    expect(find.text('STARTA TEST'), findsOneWidget);
  });

  testWidgets('modellvalet byggs från serverns katalog', (tester) async {
    await tester.pumpWidget(lab(catalog));
    await tester.pumpAndSettle();

    await tester.tap(find.text('Qwen3-ASR 0.6B'));
    await tester.pumpAndSettle();
    expect(find.text('Vosk Small Svenska 0.15'), findsOneWidget);
    expect(
        find.text('Vosk Small English 0.15 (ej installerad)'), findsOneWidget);
  });

  testWidgets('Vosk begränsar språkvalet till modellens språk', (tester) async {
    await tester.pumpWidget(lab(catalog));
    await tester.pumpAndSettle();
    expect(find.text('Svenska'), findsOneWidget);
    expect(find.text('Auto'), findsNothing);

    await tester.tap(find.text('Qwen3-ASR 0.6B'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Vosk Small Svenska 0.15').last);
    await tester.pumpAndSettle();

    await tester.tap(find.text('Svenska'));
    await tester.pumpAndSettle();
    expect(find.text('English'), findsNothing);
  });

  testWidgets('språket byts när modellen inte stöder det valda',
      (tester) async {
    await tester.pumpWidget(lab(catalog));
    await tester.pumpAndSettle();

    // Pick English on a multilingual model, then a Swedish-only checkpoint.
    await tester.tap(find.text('Svenska'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('English').last);
    await tester.pumpAndSettle();
    expect(find.text('English'), findsOneWidget);

    await tester.tap(find.text('Qwen3-ASR 0.6B'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Vosk Small Svenska 0.15').last);
    await tester.pumpAndSettle();

    expect(find.text('Svenska'), findsOneWidget);
    expect(find.text('English'), findsNothing);
  });

  testWidgets('tom katalog låser startknappen', (tester) async {
    await tester.pumpWidget(lab(const []));
    await tester.pumpAndSettle();

    expect(find.text('Servern erbjuder inga modeller'), findsOneWidget);
    final button = tester.widget<FilledButton>(find.byType(FilledButton));
    expect(button.onPressed, isNull);
  });

  testWidgets('talarbenchmark visar modellstatus och kräver inspelning',
      (tester) async {
    final client = MockClient((request) async {
      final body = switch (request.url.path) {
        '/recordings' || '/speaker-profiles' => const [],
        '/speaker-models' => const [
            {'id': 'campplus', 'name': 'CAM++ 200k', 'available': true},
            {'id': 'eres2net', 'name': 'ERes2Net 200k', 'available': false}
          ],
        '/models' => const [
            {
              'id': 'vosk-sv',
              'name': 'Vosk Svenska',
              'available': true,
              'languages': ['sv']
            }
          ],
        _ => throw StateError('Unexpected ${request.url.path}')
      };
      return http.Response(jsonEncode(body), 200,
          headers: {'content-type': 'application/json; charset=utf-8'});
    });
    await tester
        .pumpWidget(MaterialApp(home: SpeakerBenchmarkPage(client: client)));
    await tester.pumpAndSettle();

    expect(find.text('Vem sade vad?'), findsOneWidget);
    expect(find.text('IDENTIFIERING'), findsOneWidget);
    expect(find.text('CAM++ 200k'), findsOneWidget);
    await tester.drag(find.byType(ListView), const Offset(0, -700));
    await tester.pumpAndSettle();
    expect(find.text('Runtime saknas'), findsWidgets);
    final start = tester.widget<FilledButton>(
        find.widgetWithText(FilledButton, 'KÖR ALLA VALDA'));
    expect(start.onPressed, isNull);
  });

  testWidgets('profilvyn visar tomt röstarkiv', (tester) async {
    final client = MockClient((request) async => http.Response('[]', 200,
        headers: {'content-type': 'application/json'}));
    await tester
        .pumpWidget(MaterialApp(home: SpeakerProfilesPage(client: client)));
    await tester.pumpAndSettle();

    expect(find.text('Röstarkiv'), findsOneWidget);
    expect(find.text('Inga röster registrerade ännu.'), findsOneWidget);
    expect(find.text('NY PROFIL'), findsOneWidget);
  });
}
