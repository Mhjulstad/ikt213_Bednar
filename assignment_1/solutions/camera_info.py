import cv2
import os


def save_camera_info():
    # Open the default camera
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    # Get camera properties
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cam.get(cv2.CAP_PROP_FPS)

    save_path = os.path.expanduser("~/Semester 5/ikt213g26h/IKT213_Bednar/assignment_1/solutions/camera_outputs.txt")

    with open(save_path, "w") as f:
        f.write(f"fps: {frame_fps:.0f}\n")
        f.write(f"width: {frame_width}\n")
        f.write(f"height: {frame_height}\n")

    print(f"Camera information saved to {save_path}")

    cam.release()


def main():
    save_camera_info()

if __name__ == "__main__":
    main()