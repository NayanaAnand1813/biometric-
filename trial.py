import cv2
import os
import numpy as np

# Create faces directory
os.makedirs("faces", exist_ok=True)
MODEL_FILE = "trainer.yml"


# Step 1: Register a new face (No manual ESC required)
def capture_faces():
    person_name = input("Enter the person's name: ").strip()

    if not person_name:
        print("Name cannot be empty.")
        return

    person_path = os.path.join("faces", person_name)
    os.makedirs(person_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    if face_cascade.empty():
        print("Error: Face detection file could not be loaded.")
        return

    # Using CAP_DSHOW for fast startup on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\nLook at the camera and move your head slightly...")
    print("Capturing will stop automatically when 50 images are collected.\n")

    while count < 50:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            count += 1
            file_path = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)

            # Draw visual feedback
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"Captured: {count}/50", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            # Break immediately out of the detection loop if target reached
            if count >= 50:
                break

        cv2.imshow("Face Registration - Automatic Capturing", frame)

        # Still allows user to abort early via ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            print("\nRegistration cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nSuccessfully captured {count} images for {person_name}.")
    
    # Proactively trigger a model update so the new user can be recognized
    if count >= 50:
        train_model()


# Internal Helper: Train the model and save to disk
def train_model():
    print("\nUpdating face recognition database. Please wait...")
    train_dir = "faces"
    training_faces = []
    labels = []
    label_dict = {}

    if not os.path.exists(train_dir):
        return None, None

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
        return None, None

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("Install OpenCV contrib using: pip install opencv-contrib-python")
        return None, None

    recognizer.train(training_faces, np.array(labels, dtype=np.int32))
    recognizer.write(MODEL_FILE)
    
    # Also save labels mapping to a small text file alongside the model
    with open("labels.txt", "w") as f:
        for lbl, name in label_dict.items():
            f.write(f"{lbl}:{name}\n")

    print(f"Database optimized! Registered {len(label_dict)} user(s) across {len(training_faces)} images.")
    return recognizer, label_dict


# Step 2: Recognize faces using saved model configurations
def recognize_faces():
    # Attempt to load names dictionary from text file
    label_dict = {}
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r") as f:
            for line in f:
                if ":" in line:
                    lbl, name = line.strip().split(":", 1)
                    label_dict[int(lbl)] = name

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("Install OpenCV contrib using: pip install opencv-contrib-python")
        return

    # Read trained model configuration if it exists, otherwise build it now
    if os.path.exists(MODEL_FILE) and label_dict:
        recognizer.read(MODEL_FILE)
    else:
        print("No pre-trained database found.")
        recognizer, label_dict = train_model()
        if recognizer is None:
            print("Please register a face first.")
            return

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Error: Face detection file could not be loaded.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\nStarting face recognition...")
    print("Press ESC while focusing on the camera window to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            label, confidence = recognizer.predict(face)

            # Lower confidence numbers = closer match in LBPH
            if confidence < 75:
                name = label_dict.get(label, "Unknown Mapping")
                text = f"{name} ({confidence:.1f})"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
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
1. Register New Face (Auto-Capture 50 images)
2. Recognize Face (Fast startup via saved database)
3. List Registered Users
4. Force Model Re-train
5. Exit
""")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            capture_faces()
        elif choice == "2":
            recognize_faces()
        elif choice == "3":
            list_registered_users()
        elif choice == "4":
            train_model()
        elif choice == "5":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
