import cv2
from ultralytics import YOLO
import vehicles

# Load YOLOv8 model
model = YOLO('yolov8s.pt')  # Ensure you have the model file in the working directory

# Define video source and variables
cap = cv2.VideoCapture("Videos/video1.mp4")
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
max_p_age = 5
pid = 1
cnt_up = 0
cnt_down = 0
line_up = 400
line_down = 250
up_limit = 230
down_limit = int(4.5 * (500 / 5))

# Initialize counters for individual vehicle types
bicycle_count = 0
car_count = 0
motorcycle_count = 0
bus_count = 0
truck_count = 0

print("Vehicle Detection, Classification, and Counting using YOLOv8")

# Video Processing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame = cv2.resize(frame, (900, 500))  # Resize frame to a consistent size

    # YOLOv8 Inference
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data is not None else []

    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection[:6]

        # Filter detections for COCO vehicle classes: Bicycle, Car, Motorcycle, Bus, Truck
        if int(class_id) in [1, 2, 3, 5, 7]:  # 1: Bicycle, 2: Car, 3: Motorcycle, 5: Bus, 7: Truck
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            w = int(x2 - x1)
            h = int(y2 - y1)

            new = True
            if cy in range(up_limit, down_limit):
                for i in cars:
                    if abs(cx - i.getX()) <= w and abs(cy - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.going_UP(line_down, line_up):
                            cnt_up += 1
                        elif i.going_DOWN(line_down, line_up):
                            cnt_down += 1

                        break

                if new:
                    p = vehicles.Car(pid, cx, cy, max_p_age)
                    cars.append(p)
                    pid += 1

                    # Update individual vehicle counters
                    if class_id == 1:  # Bicycle
                        bicycle_count += 1
                    elif class_id == 2:  # Car
                        car_count += 1
                    elif class_id == 3:  # Motorcycle
                        motorcycle_count += 1
                    elif class_id == 5:  # Bus
                        bus_count += 1
                    elif class_id == 7:  # Truck
                        truck_count += 1

            # Draw bounding box and classify vehicles based on YOLO's class ID
            label = (
                "Bicycle" if class_id == 1 else
                "Car" if class_id == 2 else
                "Motorcycle" if class_id == 3 else
                "Bus" if class_id == 5 else
                "Truck"
            )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw lines and counters
    str_up = f'UP: {cnt_up}'
    str_down = f'DOWN: {cnt_down}'
    frame = cv2.line(frame, (0, line_up), (900, line_up), (255, 0, 255), 3, 8)
    frame = cv2.line(frame, (0, up_limit), (900, up_limit), (0, 255, 255), 3, 8)
    frame = cv2.line(frame, (0, down_limit), (900, down_limit), (255, 0, 0), 3, 8)
    frame = cv2.line(frame, (0, line_down), (900, line_down), (255, 0, 0), 3, 8)

    # Display individual vehicle counts
    cv2.putText(frame, f'Bicycles: {bicycle_count}', (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Cars: {car_count}', (10, 60), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Motorcycles: {motorcycle_count}', (10, 80), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Buses: {bus_count}', (10, 100), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Trucks: {truck_count}', (10, 120), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the live frame
    cv2.imshow('Vehicle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
