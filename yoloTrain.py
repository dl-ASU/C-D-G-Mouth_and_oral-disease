import argparse
import torch
import cv2
import matplotlib.pyplot as plt
import glob
import os


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument('--data', type=str, required=True, help=r"Path to the dataset.yaml file")
    parser.add_argument('--version', type=str, default='yolov8n.pt', choices=['yolov5s.pt', 'yolov8n.pt', 'yolov11n.pt'],
                        help="YOLO version to use: yolov5s.pt, yolov8n.pt, or yolov11n.pt")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--img-size', type=int, default=640, help="Image size for training")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--workers', type=int, default=8, help="Number of data loading workers")
    parser.add_argument('--project', type=str, default='runs/train', help="Directory to save the results")
    parser.add_argument('--val', action='store_true', help="Validate after training")
    parser.add_argument('--predict', action='store_true', help="Run predictions after training")
    parser.add_argument('--augment', action='store_true', help="Use data augmentation")
    parser.add_argument('--weight-decay', type=float, default=0.0005, help="Weight decay for regularization")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count()

    print(f"Using device: {device}, Number of GPUs: {num_gpus}")
    # Loading model based on YOLO version
    try:
        from ultralytics import YOLO
        model = YOLO(args.version)

    except ImportError:
        print("Ultralytics YOLO module not found. Please ensure it is installed.")
        return

    if num_gpus > 1:
        model = model.autoscale()
    else:
        model = model.to(device)

    # Train the model
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        lr0=args.lr,
        device=device,
        workers=args.workers,
        project=args.project,
        augment=args.augment,
        weight_decay=args.weight_decay,
        patience=args.patience
    )

    # Validation after training if specified
    if args.val:
        model.val(data=args.data)

    # Predict after training if specified
    if args.predict:
        output_dir = os.path.join(args.project, 'output', 'predict')
        os.makedirs(output_dir, exist_ok=True)

        model.predict(source=args.data, save=True)

        # Displaying predictions
        display_predictions(output_dir)


def display_predictions(output_dir):
    output_images = os.path.join(output_dir, '*.jpg')

    # Read and display each output image
    for img_path in glob.glob(output_images):
        # Read the image
        img = cv2.imread(img_path)

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Predictions on {img_path}")
        plt.show()


if __name__ == '__main__':
    main()


# test using yolo8 with val and predict
'''
python main.py --data "C:\\Users\\zyn66\\Downloads\\Oral Cancer.v8i.yolov5pytorch\\data.yaml" --version yolov8n.pt --val --predict --epochs 1 --batch-size 16 --img-size 640 --lr 0.01 --workers 8 --augment --weight-decay 0.0005 --patience 5 --project runs1/train

'''
