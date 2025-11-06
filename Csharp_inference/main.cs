using OpenCvSharp;
using System.Diagnostics;
using ColonyCounter; 

// =============================
// (1) 설정값
// =============================
string onnxModelPath = @"C:\workspace\ColonyDetectionCS\colony_model_opset12.onnx";
string datasetRoot = @"C:\workspace\datasets\colony_2class_noval\images";
float overlapRatio = 0.2f;

Console.WriteLine("C# Colony Counter 시작...");
var totalSw = Stopwatch.StartNew(); // 전체 시간 측정

// =============================
// (2) ColonyDetector 클래스 생성 (모델 로드)
// =============================
ColonyDetector detector;
try
{
    detector = new ColonyDetector(onnxModelPath); // <- 인스턴스 생성 시 모델 로드
    Console.WriteLine($"ONNX 모델 로드 성공: {onnxModelPath}");
}
catch (Exception ex)
{
    Console.WriteLine($"모델 로드 실패: {ex.Message}");
    return; // 프로그램 종료
}

// =============================
// (3) 이미지 파일 목록 가져오기
// =============================
var imageDirs = new[] { Path.Combine(datasetRoot, "test") };
var imageFiles = new List<string>();
foreach (var dir in imageDirs)
{
    if (!Directory.Exists(dir)) continue;
    imageFiles.AddRange(Directory.EnumerateFiles(dir, "*.png", SearchOption.TopDirectoryOnly));
    imageFiles.AddRange(Directory.EnumerateFiles(dir, "*.jpg", SearchOption.TopDirectoryOnly));
}
imageFiles.Sort();
Console.WriteLine($"Total Images Found: {imageFiles.Count}");

// =============================
// (4) 메인 루프 
// =============================
int count = 0;
foreach (var imgPath in imageFiles)
{
    count++;
    var imgSw = Stopwatch.StartNew();
    Console.WriteLine($"[{count}/{imageFiles.Count}] Processing: {Path.GetFileName(imgPath)}");

    using Mat img = Cv2.ImRead(imgPath, ImreadModes.Color);
    if (img.Empty())
    {
        Console.WriteLine("  -> Image load failed, skipping.");
        continue;
    }

    // 1) Full Prediction (1x1) - (Single inference 확인용, 필요하면 주석해제)
    //List<Prediction> full_preds = detector.RunFullPrediction(img);
    //Console.WriteLine($"  -> Found {full_preds.Count} Full (1x1) boxes");

    // 2) Multi-Scale Prediction (MS) - (MS 결과 확인용, 필요하면 주석해제)
    //List<Prediction> ms_preds = detector.RunMultiScalePrediction(img, overlapRatio);
    //Console.WriteLine($"  -> Found {ms_preds.Count} Multi-Scale boxes");

    // 3) Isolated Colony Prediction (MS 이후 Isolated colony filtering)
    List<Prediction> iso_preds = detector.RunIsolatedPrediction(img, overlapRatio);
    Console.WriteLine($"  -> Found {iso_preds.Count} Isolated boxes");

    imgSw.Stop();
    Console.WriteLine($"  -> Done in {imgSw.ElapsedMilliseconds} ms");
}

// =============================
// (5) 종료
// =============================
detector.Dispose(); // 세션 해제
totalSw.Stop();
Console.WriteLine($"완료! (Total Time: {totalSw.Elapsed.TotalSeconds:F2}s)");
// (결과 폴더 메시지 삭제)
Console.ReadLine();
