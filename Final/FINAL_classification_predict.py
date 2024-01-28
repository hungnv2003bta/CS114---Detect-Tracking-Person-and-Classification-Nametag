from torchvision import transforms
from PIL import Image
import torch
from torchvision import models, transforms
import cv2

class ImagePredictor:
    def __init__(self, model_path="model/28.1.model_ft-VER6.pt", class_names=["without_name_tag", "with_name_tag"]):
        self.model = self.load_model(model_path)
        self.class_names = class_names

        # Custom transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def load_model(self, model_path):
        model_conv = models.resnet18()
        # Load the entire model and check if it's a state dictionary
        model_conv  = torch.load(model_path, map_location=torch.device('cpu'))
        device = torch.device("cpu")
        model_conv = model_conv.to(device)
        model_conv .eval()
        return model_conv

    def preprocess_image(self, image):
        # # Open the image and convert to RGB
        # image = Image.open(image).convert('RGB')
        # image = self.transform(image)  # Apply the specified transform
        # return image

        #   Convert NumPy array to PIL Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)  # Apply the specified transform
        return image


    def predict_image(self, image):
        # Preprocess the image
        image = self.preprocess_image(image)

        # Add the batch dimension if your model expects a batch
        image = image.unsqueeze(0)

        # Move the image to the appropriate device (CPU or GPU)
        device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else "cpu"
        image = image.to(device)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(image)

        # Convert the model outputs to probabilities using softmax
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

        return self.class_names[probs.argmax().item()]
        # # Display the image and predicted probabilities
        # plt.imshow(np.array(Image.open(image_path)))
        # plt.title(f'Predicted class: {self.class_names[probs.argmax().item()]}, Probability: {probs.max().item():.4f}')
        # plt.show()

# image_predictor = ImagePredictor()
# # To predict on an image, we have change some parameter at process_image
# x = image_predictor.predict_image('/Users/hungnguyen/Projects/python/yolov8/results_without_name_tag/28_01_2024_16_03_00.jpg')
# print(x)