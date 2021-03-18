import numpy as np
import torch
import cv2 as cv
import  torch.nn as nn



class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

cap = cv.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    imgG = cv.GaussianBlur(gray, (5,5),0)
    erosion = cv.erode(imgG,(3,3),iterations=3)
    dilate = cv.dilate(erosion,(3,3),iterations=3)
    edged = cv.Canny(dilate,80,200,255)
    contours,hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    digitcnts = []
    for i in contours:
        (x,y,w,h) = cv.boundingRect(i)
        if w < 100 and h < 160:
            digitcnts.append(i)
    m = 0
    for c in digitcnts:
        (x,y,w,h) = cv.boundingRect(c)
        m += 1
        roi1 = frame[y:y + h, x:x + w]
        height, width, channel = roi1.shape
        for i in range(height):
            for j in range(width):
                b, g, r = roi1[i, j]
                if g > 180:
                    b = 255
                    r = 255
                    g = 255
                else :
                    b = 0
                    g = 0
                    r = 0
                roi1[i, j] = [b, g, r]
        roi1 = 255 - roi1
        roi2 = cv.copyMakeBorder(roi1, 30,30, 30, 30, cv.BORDER_CONSTANT, value=[0, 0, 0])
        cv.imwrite('%d.png' %m, roi2)
        img1 = cv.imread('%d.png' %m, 0)
        img1 = cv.GaussianBlur(img1, (5,5), 0)

        img1 = cv.dilate(img1, (3,3), iterations=3)

        img2 = cv.resize(img1, (28,28),interpolation=cv.INTER_CUBIC)
        img3 = np.array(img2) / 255
        img4 = np.reshape(img3, [-1,784])

        images = torch.tensor(img4, dtype=torch.float32)
        images = images.resize(1, 1, 28, 28)

        model = ConvNet()
        model.load_state_dict(torch.load('model.ckpt'))
        model.eval()
        outputs = model(images)
        values, indices = outputs.max(1)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 0)
        cv.putText(frame, str(indices[0]), (x, y), font, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow("capture", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

