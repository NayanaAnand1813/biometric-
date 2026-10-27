import cv2
import os
import numpy as np

# Create faces directory
os.makedirs("faces", exist_ok=True)


# Step 1: Register a new face
def capture_faces():
    person_name = input("Enter the person's name: ").strip()

    if not person_name:
        print("Name cannot be empty.")
        return

    person_path = os.path.join("faces", person_name)
    os.makedirs(person_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("Error: Face detection file could not be loaded.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\nLook at the camera.")
    print("Press ESC to stop capturing.\n")

    while count < 50:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(
            gray, 1.3, 5
        )

        for (x, y, w, h) in detected_faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            count += 1
            file_path = os.path.join(
                person_path, f"{count}.jpg"
            )
            cv2.imwrite(file_path, face)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"Images: {count}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2
            )

            if count >= 50:
                break

        cv2.imshow("Face Registration", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nCaptured {count} images for {person_name}.")


# Step 2: Train the model and recognize faces
def recognize_faces():
    train_dir = "faces"
    training_faces = []
    labels = []
    label_dict = {}

    if not os.path.exists(train_dir):
        print("No registered faces found.")
        return

    # Load registered face images
    for person_name in sorted(os.listdir(train_dir)):
        person_path = os.path.join(train_dir, person_name)

        if not os.path.isdir(person_path):
            continue

        person_images = []

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (200, 200))
                person_images.append(img)

        if person_images:
            label = len(label_dict)
            label_dict[label] = person_name

            for img in person_images:
                training_faces.append(img)
                labels.append(label)

    if not training_faces:
        print("No training images found. Register a face first.")
        return

    # Train LBPH face recognizer
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("Install OpenCV contrib using:")
        print("pip install opencv-contrib-python")
        return

    recognizer.train(
        training_faces,
        np.array(labels, dtype=np.int32)
    )

    print(f"\nTraining completed with {len(training_faces)} images.")
    print("Starting face recognition...")
    print("Press ESC to exit.\n")

    face_cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
)

    if face_cascade.empty():
        print("Error: Face detection file could not be loaded.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(
            gray, 1.3, 5
        )

        for (x, y, w, h) in detected_faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            label, confidence = recognizer.predict(face)

            if confidence < 80:
                name = label_dict[label]
                text = f"{name} ({confidence:.1f})"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                color, 2
            )

            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# Step 3: List registered users
def list_registered_users():
    if not os.path.exists("faces"):
        print("No registered users found.")
        return

    users = [
        name for name in os.listdir("faces")
        if os.path.isdir(os.path.join("faces", name))
        and any(
            file.lower().endswith((".jpg", ".jpeg", ".png"))
            for file in os.listdir(os.path.join("faces", name))
        )
    ]

    if not users:
        print("No registered users found.")
        return

    print("\nRegistered Users:")
    for i, name in enumerate(sorted(users), start=1):
        print(f"{i}. {name}")


# Main menu
def main():
    while True:
        print("""
===== FACE RECOGNITION SYSTEM =====

1. Register New Face
2. Recognize Face
3. List Registered Users
4. Exit
""")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            capture_faces()
        elif choice == "2":
            recognize_faces()
        elif choice == "3":
            list_registered_users()
        elif choice == "4":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()