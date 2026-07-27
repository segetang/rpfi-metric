"""
roi_extract.py -- 비디오에서 얼굴 ROI를 검출하고 프레임별 평균 RGB 시계열을 만든다.

원 추출 파이프라인(remotebiosensing/rppg)의 CONT 경로
(`rppg/preprocessing/image_preprocess.py`의 `CONT_preprocess_Video`)를 재현한 참조
구현이다. 논문에서 사용한 extractor trace는 이 파이프라인으로 미리 뽑아 둔 .npy
아카이브이며, 원본 추출 스크립트 자체는 본 저장소에 포함되어 있지 않다.

CONT 경로 요약
--------------
  1) 1차 패스: 전 프레임에 face_recognition.face_locations(frame, 1) 실행 (1 = CNN 모델)
  2) 박스 → 정사각 크롭: 세로 20% 확장 후 bbox_size = 1.2 * (maxy - miny)
  3) 2차 패스: 저장된 박스로 크롭 → img_size x img_size 리사이즈 (INTER_AREA)
  4) 영상 전체 정규화: (x - mean) / std,  반환 shape = (T, img_size, img_size, 3)

주의
----
ROI/스킨 분할 설정은 파이프라인마다 다르므로, 이 스크립트로 다시 뽑은 trace는
아카이브된 .npy와 비트 단위로 일치하지 않는다.

사용법
------
    from roi_extract import extract_roi_video, rgb_series
    frames = extract_roi_video('subject.avi', img_size=128)   # (T, S, S, 3)
    rgb    = rgb_series(frames)                               # (T, 3)
"""
import numpy as np

BBOX_COEF = 1.2      # CONT 경로 기본 확대 계수 (.mat 입력은 1.5)
Y_EXT = 0.2          # 세로 20% 확장


def _face_box(face_location):
    """face_recognition의 (top, right, bottom, left) -> 정사각 크롭 박스.

    CONT_preprocess_Video와 동일한 산식:
        miny -= (maxy - miny) * 0.2
        bbox_size = 1.2 * (maxy - miny)
    """
    lm_x = face_location[1::2]          # right, left
    lm_y = face_location[0::2]          # top, bottom
    minx, maxx = min(lm_x), max(lm_x)
    miny, maxy = min(lm_y), max(lm_y)
    cnt_x, cnt_y = (minx + maxx) / 2, (miny + maxy) / 2
    miny = miny - (maxy - miny) * Y_EXT          # 세로 확장
    bbox_size = int(BBOX_COEF * (maxy - miny))
    x0 = int(cnt_x - bbox_size / 2)
    y0 = int(cnt_y - bbox_size / 2)
    return x0, y0, bbox_size


def _crop(frame, x0, y0, size):
    """np.take + clipping으로 경계를 벗어나는 박스도 안전하게 크롭."""
    h, w = frame.shape[:2]
    ys = np.clip(np.arange(y0, y0 + size), 0, h - 1)
    xs = np.clip(np.arange(x0, x0 + size), 0, w - 1)
    return frame[np.ix_(ys, xs)]


def extract_roi_video(path, img_size=128, model=1, normalize=True):
    """비디오 -> (T, img_size, img_size, 3) 얼굴 ROI 스택.

    Args:
        path: 비디오 파일 경로.
        img_size: 크롭 리사이즈 한 변의 길이.
        model: face_recognition 모델 (1 = CNN, 0 = HOG). CONT 경로 기본값은 1.
        normalize: True이면 영상 전체에 (x - mean) / std 적용.

    Returns:
        (T, img_size, img_size, 3) float32 배열.
    """
    import cv2
    try:
        import face_recognition
    except ImportError as e:
        raise ImportError(
            "face_recognition 패키지가 필요하다 (pip install face_recognition). "
            "원 파이프라인의 CONT 경로가 이 검출기를 사용한다."
        ) from e

    # --- 1차 패스: 프레임별 얼굴 검출 -------------------------------------
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f'비디오를 열 수 없다: {path}')
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
        raise ValueError(f'프레임을 읽지 못했다: {path}')

    # 검출 실패 프레임은 직전 성공 박스로 채움 (전부 실패면 전체 프레임 사용)
    last = None
    for i, b in enumerate(boxes):
        if b is None:
            boxes[i] = last
        else:
            last = b
    if boxes[0] is None:
        first = next((b for b in boxes if b is not None), None)
        boxes = [first] * n if first else boxes

    # --- 2차 패스: 크롭 + 리사이즈 ---------------------------------------
    cap = cv2.VideoCapture(path)
    out = np.zeros((n, img_size, img_size, 3), dtype=np.float32)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if boxes[i] is None:                       # 전 프레임 검출 실패 시 원본 사용
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
    """(T, H, W, 3) ROI 스택 -> (T, 3) 프레임별 평균 RGB.

    POS/CHROM 등 비지도 extractor의 입력이 되는 시계열이다.
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f'(T, H, W, 3) 형태여야 한다. 입력: {frames.shape}')
    return frames.reshape(frames.shape[0], -1, 3).mean(axis=1)


if __name__ == '__main__':
    # 합성 데이터로 박스 산식과 rgb_series만 점검 (비디오/검출기 불필요)
    loc = (100, 200, 220, 80)                      # top, right, bottom, left
    x0, y0, size = _face_box(loc)
    print(f'검출 박스 (t,r,b,l)={loc} -> 크롭 x0={x0}, y0={y0}, size={size}')
    assert size == int(1.2 * ((220 - 100) * 1.2)), '박스 산식 불일치'

    dummy = np.stack([np.full((16, 16, 3), v, dtype=np.float32) for v in (1., 2., 3.)])
    rgb = rgb_series(dummy)
    print('rgb_series shape:', rgb.shape, '값:', rgb[:, 0])
    assert rgb.shape == (3, 3) and np.allclose(rgb[:, 0], [1., 2., 3.])
    print('roi_extract 자체 점검 통과')
