"""
roi_extract.py -- detect the face ROI in a video and build the per-frame mean RGB time series.

Reference implementation reproducing the CONT path of the original extraction pipeline (remotebiosensing/rppg)
(`CONT_preprocess_Video` in `rppg/preprocessing/image_preprocess.py`). The extractor traces used in the paper
are a .npy archive extracted beforehand with that pipeline; the original extraction script itself is not
included in this repository.

CONT path summary
--------------
  1) first pass: face_recognition.face_locations(frame, 1) on every frame (1 = CNN model)
  2) box -> square crop: extend vertically by 20%, then bbox_size = 1.2 * (maxy - miny)
  3) second pass: crop with the stored boxes -> resize to img_size x img_size (INTER_AREA)
  4) whole-video normalization: (x - mean) / std, returned shape = (T, img_size, img_size, 3)

Note
----
ROI/skin-segmentation settings differ between pipelines, so traces re-extracted with this script are not
bit-identical to the archived .npy files.

Usage
------
    from roi_extract import extract_roi_video, rgb_series
    frames = extract_roi_video('subject.avi', img_size=128)   # (T, S, S, 3)
    rgb    = rgb_series(frames)                               # (T, 3)
"""
import numpy as np

BBOX_COEF = 1.2      # default enlargement factor of the CONT path (1.5 for .mat inputs)
Y_EXT = 0.2          # vertical extension by 20%


def _face_box(face_location):
    """face_recognition (top, right, bottom, left) -> square crop box.

    Same formula as CONT_preprocess_Video:
        miny -= (maxy - miny) * 0.2
        bbox_size = 1.2 * (maxy - miny)
    """
    lm_x = face_location[1::2]          # right, left
    lm_y = face_location[0::2]          # top, bottom
    minx, maxx = min(lm_x), max(lm_x)
    miny, maxy = min(lm_y), max(lm_y)
    cnt_x, cnt_y = (minx + maxx) / 2, (miny + maxy) / 2
    miny = miny - (maxy - miny) * Y_EXT          # vertical extension
    bbox_size = int(BBOX_COEF * (maxy - miny))
    x0 = int(cnt_x - bbox_size / 2)
    y0 = int(cnt_y - bbox_size / 2)
    return x0, y0, bbox_size


def _crop(frame, x0, y0, size):
    """Crop safely with np.take + clipping, even for boxes that leave the frame."""
    h, w = frame.shape[:2]
    ys = np.clip(np.arange(y0, y0 + size), 0, h - 1)
    xs = np.clip(np.arange(x0, x0 + size), 0, w - 1)
    return frame[np.ix_(ys, xs)]


def extract_roi_video(path, img_size=128, model=1, normalize=True):
    """Video -> (T, img_size, img_size, 3) stack of face ROIs.

    Args:
        path: video file path.
        img_size: side length of the resized crop.
        model: face_recognition model (1 = CNN, 0 = HOG). The CONT path default is 1.
        normalize: if True, apply (x - mean) / std over the whole video.

    Returns:
        (T, img_size, img_size, 3) float32 array.
    """
    import cv2
    try:
        import face_recognition
    except ImportError as e:
        raise ImportError(
            "the face_recognition package is required (pip install face_recognition). "
            "The CONT path of the original pipeline uses this detector."
        ) from e

    # --- first pass: per-frame face detection -------------------------------------
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f'cannot open video: {path}')
    boxes, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model)
        boxes.append(_face_box(locs[0]) if locs else None)
        n += 1
    cap.release()
    if n == 0:
        raise ValueError(f'no frames could be read: {path}')

    # frames where detection failed get the last successful box (whole frame if all failed)
    last = None
    for i, b in enumerate(boxes):
        if b is None:
            boxes[i] = last
        else:
            last = b
    if boxes[0] is None:
        first = next((b for b in boxes if b is not None), None)
        boxes = [first] * n if first else boxes

    # --- second pass: crop + resize ---------------------------------------
    cap = cv2.VideoCapture(path)
    out = np.zeros((n, img_size, img_size, 3), dtype=np.float32)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if boxes[i] is None:                       # detection failed on all frames: use the original frame
            face = rgb
        else:
            x0, y0, size = boxes[i]
            face = _crop(rgb, x0, y0, size)
        out[i] = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)
    cap.release()

    if normalize:
        out = (out - out.mean()) / (out.std() + 1e-8)
    return out


def rgb_series(frames):
    """(T, H, W, 3) ROI stack -> (T, 3) per-frame mean RGB.

    This is the time series fed to unsupervised extractors such as POS/CHROM.
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f'expected shape (T, H, W, 3); got {frames.shape}')
    return frames.reshape(frames.shape[0], -1, 3).mean(axis=1)


if __name__ == '__main__':
    # check only the box formula and rgb_series on synthetic data (no video/detector needed)
    loc = (100, 200, 220, 80)                      # top, right, bottom, left
    x0, y0, size = _face_box(loc)
    print(f'detected box (t,r,b,l)={loc} -> crop x0={x0}, y0={y0}, size={size}')
    assert size == int(1.2 * ((220 - 100) * 1.2)), 'box formula mismatch'

    dummy = np.stack([np.full((16, 16, 3), v, dtype=np.float32) for v in (1., 2., 3.)])
    rgb = rgb_series(dummy)
    print('rgb_series shape:', rgb.shape, 'values:', rgb[:, 0])
    assert rgb.shape == (3, 3) and np.allclose(rgb[:, 0], [1., 2., 3.])
    print('roi_extract self-check passed')
