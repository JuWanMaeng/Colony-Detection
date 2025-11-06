using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp.Dnn;

// (1) 
namespace ColonyCounter
{
    // =============================
    // 데이터 모델 (Structs / Records)
    // =============================
    public record StageConfig(
        int Rows,
        int Cols,
        float Conf,
        float Iou,
        int MaxDet
    );

    public record Prediction(
        int ClassId,    // cls
        float X1,       // x1
        float Y1,       // y1
        float X2,       // x2
        float Y2,       // y2
        float Confidence  // conf
    )
    {
        public RectF BoxF => RectF.FromPythonBox(X1, Y1, X2, Y2);
    }

    public struct RectF
    {
        public float X, Y, Width, Height;
        public RectF(float x, float y, float width, float height)
        {
            X = x; Y = y; Width = width; Height = height;
        }
        public float Left => X;
        public float Top => Y;
        public float Right => X + Width;
        public float Bottom => Y + Height;
        public float Area => Width * Height;
        public static RectF FromPythonBox(float x1, float y1, float x2, float y2)
        {
            return new RectF(x1, y1, x2 - x1, y2 - y1);
        }
    }

    // =============================
    // 유틸리티 클래스 (Helpers)
    // =============================

    public static class DetectionUtils
    {
        private static float Intersection(RectF a, RectF b)
        {
            float x1 = Math.Max(a.Left, b.Left);
            float y1 = Math.Max(a.Top, b.Top);
            float x2 = Math.Min(a.Right, b.Right);
            float y2 = Math.Min(a.Bottom, b.Bottom);
            return Math.Max(0.0f, x2 - x1) * Math.Max(0.0f, y2 - y1);
        }

        public static float IoU(RectF a, RectF b)
        {
            float inter = Intersection(a, b);
            if (inter <= 0) return 0.0f;
            float union = a.Area + b.Area - inter;
            return (union > 0) ? (inter / union) : 0.0f;
        }

        public static float IoA_Smaller(RectF a, RectF b)
        {
            float inter = Intersection(a, b);
            if (inter <= 0) return 0.0f;
            float denom = Math.Min(a.Area, b.Area);
            return (denom > 0) ? (inter / denom) : 0.0f;
        }

        public static RectF MakeSquareBox(float x1, float y1, float x2, float y2, int imgW, int imgH)
        {
            float w = x2 - x1;
            float h = y2 - y1;
            float side = Math.Max(w, h);
            float cx = (x1 + x2) / 2.0f;
            float cy = (y1 + y2) / 2.0f;
            float x1New = Math.Max(0.0f, cx - side / 2.0f);
            float y1New = Math.Max(0.0f, cy - side / 2.0f);
            float x2New = Math.Min(imgW, cx + side / 2.0f);
            float y2New = Math.Min(imgH, cy + side / 2.0f);
            return new RectF(x1New, y1New, x2New - x1New, y2New - y1New);
        }
    }

    public static class Preprocessing
    {
        public static Mat Letterbox(
            Mat im,
            OpenCvSharp.Size newShape,
            Scalar color,
            out OpenCvSharp.Size2f ratio,
            out OpenCvSharp.Point pad)
        {
            OpenCvSharp.Size shape = im.Size();
            float r = Math.Min((float)newShape.Height / shape.Height, (float)newShape.Width / shape.Width);
            ratio = new OpenCvSharp.Size2f(r, r);
            OpenCvSharp.Size newUnpad = new OpenCvSharp.Size((int)Math.Round(shape.Width * r), (int)Math.Round(shape.Height * r));
            int dw = newShape.Width - newUnpad.Width;
            int dh = newShape.Height - newUnpad.Height;
            dw /= 2;
            dh /= 2;

            Mat resizedIm;
            if (shape != newUnpad)
            {
                resizedIm = new Mat();
                Cv2.Resize(im, resizedIm, newUnpad, 0, 0, InterpolationFlags.Linear);
            }
            else
            {
                resizedIm = im.Clone();
            }

            int top = (int)Math.Round(dh - 0.1);
            int bottom = (int)Math.Round(dh + 0.1);
            int left = (int)Math.Round(dw - 0.1);
            int right = (int)Math.Round(dw + 0.1);
            Mat imOut = new Mat();
            Cv2.CopyMakeBorder(resizedIm, imOut, top, bottom, left, right, BorderTypes.Constant, color);
            resizedIm.Dispose();
            pad = new OpenCvSharp.Point(left, top);
            return imOut;
        }
    }

    // =============================
    // OnnxPredictor와 InferencePipeline의 로직을 통합한 메인 클래스.
    // =============================
    public class ColonyDetector : IDisposable
    {
        private readonly InferenceSession _session;

        // --- 상수 정의 ---
        private const int INPUT_WIDTH = 640;
        private const int INPUT_HEIGHT = 640;
        private const int NUM_CLASSES = 2;
        private const int CLASS_COLONY_ID = 0;
        private static readonly List<StageConfig> STAGE_CONFIGS = new()
        {
            new StageConfig(Rows: 1, Cols: 1, Conf: 0.5f, Iou: 0.5f, MaxDet: 3000),
            new StageConfig(Rows: 2, Cols: 2, Conf: 0.5f, Iou: 0.4f, MaxDet: 3000),
            new StageConfig(Rows: 4, Cols: 4, Conf: 0.5f, Iou: 0.3f, MaxDet: 3000)
        };
        private const float MERGE_NMS_IOU = 0.1f;
        private const int MERGE_MAX_KEEP = 5000;
        private const float ISOLATED_IOU_THR = 0.05f;
        private const float ISOLATED_IOA_THR = 0.05f;

        public ColonyDetector(string modelPath)
        {
            var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_CUDA();
            //sessionOptions.AppendExecutionProvider_CPU(); // CUDA 실패 시 
            _session = new InferenceSession(modelPath, sessionOptions);
        }

        public void Dispose()
        {
            _session?.Dispose();
        }

        // =============================
        // (1) 공개 API (Public Methods)
        // =============================

        public List<Prediction> RunFullPrediction(Mat img) // 1x1 스케일에서 전체 이미지 예측
        {
            var fullCfg = STAGE_CONFIGS[0];
            return RunSinglePrediction(img, fullCfg.Conf, fullCfg.Iou, fullCfg.MaxDet);
        }

        public List<Prediction> RunMultiScalePrediction(Mat img, float overlapRatio = 0.2f) // 다중 스케일 예측
        {
            int H = img.Height;
            int W = img.Width;
            var merged = new List<Prediction>();

            foreach (var cfg in STAGE_CONFIGS)
            {
                var gridPreds = InferGridWithOverlap(img, cfg, overlapRatio);
                merged.AddRange(gridPreds);
            }

            var sq = new List<Prediction>();
            foreach (var p in merged)
            {
                if (p.ClassId == CLASS_COLONY_ID)
                {
                    RectF sqBox = DetectionUtils.MakeSquareBox(p.X1, p.Y1, p.X2, p.Y2, W, H);
                    sq.Add(new Prediction(p.ClassId, sqBox.X, sqBox.Y, sqBox.Right, sqBox.Bottom, p.Confidence));
                }
                else
                {
                    sq.Add(p);
                }
            }
            return GlobalNms(sq, iouThresh: MERGE_NMS_IOU, maxKeep: MERGE_MAX_KEEP);
        }

        public List<Prediction> RunIsolatedPrediction(Mat img, float overlapRatio = 0.2f) // 고립된 콜로니 예측
        {
            List<Prediction> ms_preds = RunMultiScalePrediction(img, overlapRatio);
            List<Prediction> iso = FilterIsolatedBoxes(ms_preds, thrIou: ISOLATED_IOU_THR, thrIoa: ISOLATED_IOA_THR);
            return iso.Where(p => p.ClassId == CLASS_COLONY_ID).ToList();
        }


        // =============================
        // (2) 비공개 헬퍼 (Private Methods)
        // =============================

        private List<Prediction> RunSinglePrediction(Mat imgOriginal, float confThresh, float iouThresh, int maxDet) // 이미지 한장씩 처리
        {
            if (imgOriginal == null || imgOriginal.Empty())
            {
                return new List<Prediction>();
            }

            int imgOriginalW = imgOriginal.Width;
            int imgOriginalH = imgOriginal.Height;

            // 1. 전처리
            Mat imgResized = Preprocessing.Letterbox(
                imgOriginal, new OpenCvSharp.Size(INPUT_WIDTH, INPUT_HEIGHT), new Scalar(114, 114, 114),
                out OpenCvSharp.Size2f ratio, out OpenCvSharp.Point pad
            );
            Mat blob = CvDnn.BlobFromImage(
                imgResized, 1.0 / 255.0, new OpenCvSharp.Size(INPUT_WIDTH, INPUT_HEIGHT),
                Scalar.All(0), true, false
            );
            imgResized.Dispose();

            // 2. 추론
            int inputSize = 1 * 3 * INPUT_HEIGHT * INPUT_WIDTH;
            float[] inputData = new float[inputSize];
            System.Runtime.InteropServices.Marshal.Copy(blob.Data, inputData, 0, inputSize);

            var inputTensor = new DenseTensor<float>(inputData, new[] { 1, 3, INPUT_HEIGHT, INPUT_WIDTH });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_session.InputNames[0], inputTensor)
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
            var outputTensor = results[0].AsTensor<float>();

            int numDetections = outputTensor.Dimensions[2]; // 8400
            blob.Dispose();

            // 3. 후처리 (NMS)
            var boxes_xyxy_cls0 = new List<RectF>();
            var boxes_xywh_cls0 = new List<Rect>();
            var scores_cls0 = new List<float>();
            var boxes_xyxy_cls1 = new List<RectF>();
            var boxes_xywh_cls1 = new List<Rect>();
            var scores_cls1 = new List<float>();

            for (int i = 0; i < numDetections; i++)
            {
                float score_cls0 = outputTensor[0, 4, i];
                float score_cls1 = outputTensor[0, 5, i];
                float cx = outputTensor[0, 0, i];
                float cy = outputTensor[0, 1, i];
                float w = outputTensor[0, 2, i];
                float h = outputTensor[0, 3, i];
                float x1 = cx - w / 2;
                float y1 = cy - h / 2;
                float x2 = cx + w / 2;
                float y2 = cy + h / 2;

                if (score_cls0 > confThresh)
                {
                    boxes_xyxy_cls0.Add(RectF.FromPythonBox(x1, y1, x2, y2));
                    boxes_xywh_cls0.Add(new Rect((int)x1, (int)y1, (int)w, (int)h));
                    scores_cls0.Add(score_cls0);
                }
                if (score_cls1 > confThresh)
                {
                    boxes_xyxy_cls1.Add(RectF.FromPythonBox(x1, y1, x2, y2));
                    boxes_xywh_cls1.Add(new Rect((int)x1, (int)y1, (int)w, (int)h));
                    scores_cls1.Add(score_cls1);
                }
            }

            var nmsBoxes_xyxy = new List<RectF>();
            var nmsScores = new List<float>();
            var nmsClassIds = new List<int>();

            if (scores_cls0.Count > 0)
            {
                CvDnn.NMSBoxes(boxes_xywh_cls0, scores_cls0, confThresh, iouThresh, out int[] indices_cls0);
                foreach (int i in indices_cls0)
                {
                    nmsBoxes_xyxy.Add(boxes_xyxy_cls0[i]);
                    nmsScores.Add(scores_cls0[i]);
                    nmsClassIds.Add(0);
                }
            }
            if (scores_cls1.Count > 0)
            {
                CvDnn.NMSBoxes(boxes_xywh_cls1, scores_cls1, confThresh, iouThresh, out int[] indices_cls1);
                foreach (int i in indices_cls1)
                {
                    nmsBoxes_xyxy.Add(boxes_xyxy_cls1[i]);
                    nmsScores.Add(scores_cls1[i]);
                    nmsClassIds.Add(1);
                }
            }

            if (nmsBoxes_xyxy.Count == 0)
            {
                return new List<Prediction>();
            }

            var finalDetections = nmsScores
                .Select((score, index) => new { Score = score, Index = index })
                .OrderByDescending(x => x.Score)
                .Take(maxDet)
                .ToList();

            // 4. 좌표 복원
            var finalPredsList = new List<Prediction>();
            float ratioW = ratio.Width;
            float ratioH = ratio.Height;
            int padW = pad.X;
            int padH = pad.Y;

            foreach (var det in finalDetections)
            {
                int i = det.Index;
                RectF box_xywh = nmsBoxes_xyxy[i];
                int clsId = nmsClassIds[i];
                float score = nmsScores[i];
                float x1_letter = box_xywh.X;
                float y1_letter = box_xywh.Y;
                float x2_letter = box_xywh.Right;
                float y2_letter = box_xywh.Bottom;
                float x1_unpad = x1_letter - padW;
                float y1_unpad = y1_letter - padH;
                float x2_unpad = x2_letter - padW;
                float y2_unpad = y2_letter - padH;
                float x1_orig = x1_unpad / ratioW;
                float y1_orig = y1_unpad / ratioH;
                float x2_orig = x2_unpad / ratioW;
                float y2_orig = y2_unpad / ratioH;
                x1_orig = Math.Max(0.0f, x1_orig);
                y1_orig = Math.Max(0.0f, y1_orig);
                x2_orig = Math.Min(imgOriginalW, x2_orig);
                y2_orig = Math.Min(imgOriginalH, y2_orig);
                finalPredsList.Add(new Prediction(clsId, x1_orig, y1_orig, x2_orig, y2_orig, score));
            }
            return finalPredsList;
        }

        private List<Prediction> InferGridWithOverlap(Mat img, StageConfig cfg, float overlapRatio) // 그리드 기반 겹침 전치리 후 예측
        {
            int H = img.Height;
            int W = img.Width;
            var preds = new List<Prediction>();
            float cellW = (float)W / cfg.Cols;
            float cellH = (float)H / cfg.Rows;

            for (int r = 0; r < cfg.Rows; r++)
            {
                for (int c = 0; c < cfg.Cols; c++)
                {
                    int x1 = (int)(cellW * c);
                    int y1 = (int)(cellH * r);
                    int x2 = (int)(cellW * (c + 1));
                    int y2 = (int)(cellH * (r + 1));
                    int x1_ov = (int)Math.Max(0, x1 - cellW * overlapRatio);
                    int y1_ov = (int)Math.Max(0, y1 - cellH * overlapRatio);
                    int x2_ov = (int)Math.Min(W, x2 + cellW * overlapRatio);
                    int y2_ov = (int)Math.Min(H, y2 + cellH * overlapRatio);
                    Rect cropRect = new Rect(x1_ov, y1_ov, x2_ov - x1_ov, y2_ov - y1_ov);
                    using Mat crop = new Mat(img, cropRect);
                    if (crop.Empty()) continue;

                    List<Prediction> tile_preds = RunSinglePrediction(crop, cfg.Conf, cfg.Iou, cfg.MaxDet);

                    foreach (var p in tile_preds)
                    {
                        preds.Add(new Prediction(
                            p.ClassId, p.X1 + x1_ov, p.Y1 + y1_ov,
                            p.X2 + x1_ov, p.Y2 + y1_ov, p.Confidence
                        ));
                    }
                }
            }
            return preds;
        }

        private List<Prediction> GlobalNms(List<Prediction> dets, float iouThresh, int maxKeep) // 전체 박스에 대해 NMS 수행
        {
            var sortedDets = dets.OrderByDescending(d => d.Confidence).ToList();
            var kept = new List<Prediction>();
            var suppressed = new bool[sortedDets.Count];
            for (int i = 0; i < sortedDets.Count; i++)
            {
                if (suppressed[i]) continue;
                var di = sortedDets[i];
                kept.Add(di);
                if (kept.Count >= maxKeep) break;
                for (int j = i + 1; j < sortedDets.Count; j++)
                {
                    if (suppressed[j]) continue;
                    var dj = sortedDets[j];
                    if (DetectionUtils.IoU(di.BoxF, dj.BoxF) >= iouThresh)
                    {
                        suppressed[j] = true;
                    }
                }
            }
            return kept;
        }

        private List<Prediction> FilterIsolatedBoxes(List<Prediction> preds, float thrIou, float thrIoa) // 고립된 박스 필터링
        {
            var isolated = new List<Prediction>();
            for (int i = 0; i < preds.Count; i++)
            {
                var bi = preds[i];
                bool independent = true;
                for (int j = 0; j < preds.Count; j++)
                {
                    if (i == j) continue;
                    var bj = preds[j];
                    if (DetectionUtils.IoU(bi.BoxF, bj.BoxF) >= thrIou ||
                        DetectionUtils.IoA_Smaller(bi.BoxF, bj.BoxF) >= thrIoa)
                    {
                        independent = false;
                        break;
                    }
                }
                if (independent)
                {
                    isolated.Add(bi);
                }
            }
            return isolated;
        }
    }
}