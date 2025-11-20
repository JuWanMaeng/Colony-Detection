from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolo12n.yaml")   # build a new model from YAML
    model = YOLO("yolo12n.pt")   # load a pretrained model (recommended for training)
    # model = YOLO("yolo12n.yaml").load("yolo12n.pt")  # build from YAML and transfer weights


    results = model.train(
        data=r"C:\data\product2_yolo\data.yaml",
        cfg="default.yaml",
        epochs=500,
        imgsz=640,
        project="C:/workspace/experiments",   # 결과 저장할 상위 폴더
        name="product_exp1",                   # 하위 폴더 이름
        exist_ok=True                         # 이미 있으면 덮어쓰기
    )


if __name__ == "__main__":
    main()
