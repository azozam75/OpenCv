# Object Detection with OpenCV

A small learning project that uses the [OpenCV](https://opencv.org/) library to detect objects in images using pre-trained Haar cascade classifiers. The notebook [`openCV.ipynb`](openCV.ipynb) has three parts: detecting a **cat's face and eyes**, recognizing a **Russian car license plate**, and detecting a **human face**.

## Project files

| File | Purpose |
|------|---------|
| `openCV.ipynb` | The main notebook — image display, color conversion, and the three detection examples |
| `Cat.jpg` | Test image for cat face/eye detection |
| `Russian plate.jpg` | Test image for license plate recognition |
| `Face.jpg` | Test image for human face detection |
| `haarcascade_russian_plate_number.xml` | Haar cascade for Russian license plates (loaded from this folder) |
| `haarcascade_frontalface_default.xml` | Haar cascade for human faces (loaded from this folder) |

## Requirements

- Python 3
- OpenCV: `pip install opencv-python`
- Matplotlib: `pip install matplotlib` (used to display results in the plate and face sections)
- Jupyter (or VS Code with the Jupyter extension) to run the notebook

## How to run

1. Open `openCV.ipynb` in Jupyter or VS Code.
2. Run the cells from top to bottom.
3. Cells that use `cv2.imshow(...)` open a separate image window. Because of `cv2.waitKey(0)`, the window stays open until you **press any key** while it is focused — don't close it with the mouse, or the kernel may hang. Cells that use `plt.imshow(...)` display the image inline in the notebook instead.

## Part 1 — Cat face and eye detection

### Display an image

`cv2.imread("Cat.jpg")` loads the image, and `cv2.imshow(...)` shows it in a window. `cv2.waitKey(0)` waits for a key press, then `cv2.destroyAllWindows()` closes the window.

### Color conversion

OpenCV loads images in **BGR** order (not RGB). `cv2.cvtColor(img, code)` converts between color spaces — here the image is converted to **grayscale** with `cv2.COLOR_BGR2GRAY`. This matters because Haar cascade detection works on grayscale images.

### The detection pipeline

1. **Load the classifiers.** `cv2.CascadeClassifier` loads two pre-trained Haar cascades that ship with OpenCV (from `cv2.data.haarcascades`): `haarcascade_frontalcatface.xml` for cat faces and `haarcascade_eye.xml` for eyes.
2. **Convert the image to grayscale** — cascades detect on intensity, not color.
3. **Detect faces** with `detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))`, which returns a list of `(x, y, w, h)` rectangles:
   - `scaleFactor` — how much the image is shrunk at each scan pass (1.1 = 10% steps). Smaller values find more but run slower.
   - `minNeighbors` — how many overlapping candidate detections are needed to accept a result. Higher values mean fewer false positives.
   - `minSize` — ignore anything smaller than this (in pixels).
4. **Draw a blue rectangle** around each detected face with `cv2.rectangle`.
5. **Search for eyes only inside the face region** (the ROI — region of interest — `gray[y:y+h, x:x+w]`) instead of the whole image. This is faster and avoids false eye detections elsewhere.
6. **Filter out bad eye hits:** any "eye" found in the lower 40% of the face (`ey > h * 0.6`) is skipped, since real eyes are in the upper part of the face. The remaining eyes get green rectangles.
7. **Show the result.** For `Cat.jpg` the notebook detects **1 face and 2 eyes**.

## Part 2 — Russian license plate recognition

The same Haar cascade idea, applied to a licence photo, in five steps:

1. **Load the image** — `cv2.imread("Russian plate.jpg")`.
2. **Convert color formats** — two conversions this time:
   - `cv2.COLOR_BGR2RGB` → an RGB copy for display, because **matplotlib expects RGB** (showing the raw BGR image would swap red and blue);
   - `cv2.COLOR_BGR2GRAY` → a grayscale copy for detection.
3. **Load the classifier** — this cascade doesn't ship with OpenCV, so it's loaded from the local file: `cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')`.
4. **Detect** — `plate_cascade.detectMultiScale(img_gray, minSize=(20, 20))` returns the plate rectangles.
5. **Draw and show** — green rectangles (thickness 5) around each hit, displayed inline with `plt.imshow(img_rgb)`.

## Part 3 — Human face detection

The same five steps as Part 2, condensed into a single cell and pointed at different inputs:

- **Image:** `Face.jpg`
- **Classifier:** the local `haarcascade_frontalface_default.xml`
- Convert to RGB (for display) and grayscale (for detection), run `detectMultiScale(img_gray, minSize=(20, 20))`, draw green rectangles around the detected faces, and show the result inline with matplotlib.

## Ideas to extend it

- **Real-time detection** — run the face cascade on webcam frames with `cv2.VideoCapture(0)` in a loop.
- **Combine cascades** — detect eyes or smiles (`haarcascade_smile.xml`, built into OpenCV) inside the face region of `Face.jpg`, like the cat example does.
- **Tune the parameters** — experiment with `scaleFactor` and `minNeighbors` to see how they trade off detection rate against false positives.
