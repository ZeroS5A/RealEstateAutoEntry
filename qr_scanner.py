"""
二维码识别引擎
=============
集成 zxing-cpp + OpenCV QRDetector + WeChatQRCode 三级策略。

识别流程：
  1. zxing-cpp 全图扫描（原图 + CLAHE 预处理）→ 命中直接返回
  2. OpenCV QRDetector 定位 → 裁剪 → WeChatQRCode 多尺度扫描
  3. 定位失败 → 长边一半 + 中心50% 兜底裁剪 → WeChatQRCode 扫描
"""

import cv2
import numpy as np
import os
import hashlib
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class QRScanner:
    """二维码识别引擎，封装 WeChatQRCode + OpenCV QRDetector + zxing-cpp"""

    # WeChatQRCode 模型文件 MD5 校验值
    # 参考: https://github.com/jenly1314/WeChatQRCode
    _WECHAT_QR_MODEL_MD5 = {
        "detect.prototxt": "6fb4976b32695f9f5c6305c19f12537d",
        "detect.caffemodel": "238e2b2d6f3c18d6c3a30de0c31e23cf",
        "sr.prototxt": "69db99927a70df953b471daaba03fbef",
        "sr.caffemodel": "cbfcd60361a73beb8c583eea7e8e6664",
    }

    _WECHAT_MODEL_URLS = {
        "detect.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.prototxt",
        "detect.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.caffemodel",
        "sr.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.prototxt",
        "sr.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.caffemodel",
    }

    def __init__(self, model_dir: str = "wechat_models"):
        self.model_dir = model_dir
        self.qr_detector = self._init_wechat_qrcode()
        self.last_crops: list = []

    # ==================================================================
    # 引擎初始化
    # ==================================================================

    def _init_wechat_qrcode(self):
        """初始化 WeChatQRCode 引擎，自动下载模型并校验 MD5。"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        paths = {}
        for filename, url in self._WECHAT_MODEL_URLS.items():
            file_path = os.path.join(self.model_dir, filename)
            paths[filename] = file_path
            self._ensure_model(filename, file_path, url)

        try:
            return cv2.wechat_qrcode_WeChatQRCode(
                paths["detect.prototxt"], paths["detect.caffemodel"],
                paths["sr.prototxt"], paths["sr.caffemodel"]
            )
        except Exception as e:
            print(f"[WeChatQRCode] 引擎初始化失败: {e}")
            return None

    def _ensure_model(self, filename: str, file_path: str, url: str):
        """确保单个模型文件存在且完整，否则重新下载（最多重试 3 次）。"""
        need_download = False
        if not os.path.exists(file_path):
            need_download = True
        else:
            expected_md5 = self._WECHAT_QR_MODEL_MD5.get(filename)
            if expected_md5:
                with open(file_path, 'rb') as f:
                    actual_md5 = hashlib.md5(f.read()).hexdigest()
                if actual_md5 != expected_md5:
                    print(f"[WeChatQRCode] {filename} MD5 校验失败，重新下载...")
                    os.remove(file_path)
                    need_download = True

        if need_download:
            for attempt in range(3):
                try:
                    print(f"[WeChatQRCode] 下载模型: {filename} (第{attempt + 1}次)...")
                    resp = requests.get(url, timeout=30, verify=False)
                    resp.raise_for_status()
                    with open(file_path, 'wb') as f:
                        f.write(resp.content)
                    # 下载后校验
                    expected_md5 = self._WECHAT_QR_MODEL_MD5.get(filename)
                    if expected_md5:
                        with open(file_path, 'rb') as f:
                            actual_md5 = hashlib.md5(f.read()).hexdigest()
                        if actual_md5 != expected_md5:
                            print(f"[WeChatQRCode] {filename} MD5 不匹配，将重试...")
                            os.remove(file_path)
                            continue
                    break
                except Exception as e:
                    print(f"[WeChatQRCode] 下载 {filename} 失败: {e}")

    # ==================================================================
    # 图像预处理
    # ==================================================================

    def _preprocess_for_qr(self, img):
        """二维码专用 CLAHE 对比度增强。返回单通道灰度图。"""
        if img is None or img.size == 0:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    # ==================================================================
    # WeChatQRCode 识别
    # ==================================================================

    def _try_decode_wx(self, img):
        """单次 WeChatQRCode 解码。兼容 BGR 和灰度图。"""
        if self.qr_detector is None or img is None or img.size == 0:
            return None
        try:
            res, _ = self.qr_detector.detectAndDecode(img)
            if res and len(res) > 0 and res[0]:
                return res[0]
        except Exception:
            pass
        return None

    def _scan_multiscale_wx(self, img):
        """WeChatQRCode 多尺度扫描：原图 → 2x放大 → 0.5x缩小，每种都尝试原始+预处理。"""
        if img is None or img.size == 0:
            return None

        strategies = [
            (1.0, False),   # 原图
            (1.0, True),    # 原图 + CLAHE
            (2.0, False),   # 放大2倍
            (2.0, True),    # 放大2倍 + CLAHE
            (0.5, False),   # 缩小
        ]

        for scale, preprocess in strategies:
            if scale != 1.0:
                interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
                scaled = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interp)
            else:
                scaled = img

            if preprocess:
                processed = self._preprocess_for_qr(scaled)
                if processed is not None:
                    result = self._try_decode_wx(processed)
                    if result:
                        return result
            else:
                result = self._try_decode_wx(scaled)
                if result:
                    return result
        return None

    # ==================================================================
    # OpenCV QRDetector 定位
    # ==================================================================

    def _locate_qr_opencv(self, img):
        """使用 OpenCV QRCodeDetector 多尺度定位并尝试解码二维码。
        优先 detect() 纯定位，detectAndDecode() 为补充。
        返回 (裁剪区域, 是否成功定位, 解码文本或None)。
        当 detectAndDecode 直接解码成功时，调用方可跳过后续管线。"""
        if img is None or img.size == 0:
            return None, False, None

        detector = cv2.QRCodeDetector()
        h, w = img.shape[:2]

        scales = [1.0]
        if max(h, w) > 2000:
            scales.insert(0, 0.5)
        if min(h, w) < 600:
            scales.append(2.0)

        def _extract_roi(pts, scale, tag):
            pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 4:
                return None
            pts_orig = pts / scale
            x_min = int(np.floor(np.min(pts_orig[:, 0])))
            y_min = int(np.floor(np.min(pts_orig[:, 1])))
            x_max = int(np.ceil(np.max(pts_orig[:, 0])))
            y_max = int(np.ceil(np.max(pts_orig[:, 1])))
            if x_max <= x_min or y_max <= y_min:
                return None
            qr_size = max(x_max - x_min, y_max - y_min)
            pad = int(qr_size * 1)
            x1 = max(0, x_min - pad)
            y1 = max(0, y_min - pad)
            x2 = min(w, x_max + pad)
            y2 = min(h, y_max + pad)
            print(f"  [OpenCV {tag}] scale={scale} 定位成功, 二维码~{x_max - x_min}x{y_max - y_min}px, "
                  f"裁剪区域:({x1},{y1})-({x2},{y2}) 尺寸:{x2 - x1}x{y2 - y1}")
            return img[y1:y2, x1:x2]

        for scale in scales:
            scaled = img
            if scale != 1.0:
                interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
                scaled = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interp)

            # detect() — 纯定位，不解码
            try:
                found, pts = detector.detect(scaled)
                if found and pts is not None:
                    roi = _extract_roi(pts, scale, "detect")
                    if roi is not None:
                        return roi, True, None
            except Exception as e:
                print(f"  [OpenCV detect] scale={scale} 异常: {e}")

            # detectAndDecode() — 检测+解码，解码成功则直接返回文本
            try:
                data, pts, _ = detector.detectAndDecode(scaled)
                if pts is not None:
                    roi = _extract_roi(pts, scale, "detectAndDecode")
                    if roi is not None:
                        decoded = data if (data and len(data) > 0) else None
                        if decoded:
                            print(f"  [OpenCV detectAndDecode] scale={scale} 直接解码成功！")
                        return roi, True, decoded
            except Exception as e:
                print(f"  [OpenCV detectAndDecode] scale={scale} 异常: {e}")

        # 预处理兜底
        processed = self._preprocess_for_qr(img)
        if processed is not None:
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            try:
                found, pts = detector.detect(processed_bgr)
                if found and pts is not None:
                    roi = _extract_roi(pts, 1.0, "detect+预处理")
                    if roi is not None:
                        return roi, True, None
            except Exception as e:
                print(f"  [OpenCV detect+预处理] 异常: {e}")

        print("[OpenCV] 所有定位策略均未找到二维码")
        return None, False, None

    # ==================================================================
    # 兜底裁剪策略
    # ==================================================================

    def _generate_fallback_crops(self, img):
        """生成兜底裁剪：长边方向各取一半 + 全图中心 50%。"""
        h, w = img.shape[:2]
        crops = []

        if w >= h:
            crops.append(("长边左半", img[:, :w // 2]))
            crops.append(("长边右半", img[:, w // 2:]))
        else:
            crops.append(("长边上半", img[:h // 2, :]))
            crops.append(("长边下半", img[h // 2:, :]))

        crops.append(("中心50%", img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]))
        return crops

    # ==================================================================
    # zxing-cpp 识别
    # ==================================================================

    def _try_decode_zxing(self, img):
        """zxing-cpp 解码，原图失败则对 CLAHE 预处理图重试。"""
        if img is None or img.size == 0:
            return None
        try:
            import zxingcpp

            results = zxingcpp.read_barcodes(img)
            for result in results:
                if result.valid and result.text:
                    return result.text

            processed = self._preprocess_for_qr(img)
            if processed is not None:
                results = zxingcpp.read_barcodes(processed)
                for result in results:
                    if result.valid and result.text:
                        return result.text
        except Exception as e:
            print(f"[zxing-cpp] 识别异常: {e}")
        return None

    # ==================================================================
    # 主扫描入口
    # ==================================================================

    def scan(self, img):
        """
        主入口：扫描图像中的二维码，返回解码文本或 None。

        流水线：
          1. zxing-cpp 全图扫描（原图 + CLAHE）→ 命中直接返回
          2. OpenCV QRDetector 定位 → 裁剪 → WeChatQRCode 多尺度扫描
          3. 定位失败 → 长边一半 + 中心50% 兜底裁剪 → WeChatQRCode 扫描
        """
        if img is None:
            return None

        h_img, w_img = img.shape[:2]
        print("\n" + "=" * 60)
        print(f"[QR Pipeline] 输入图像尺寸: {w_img}x{h_img}")

        # ================================================================
        # Step 1: zxing-cpp 全图扫描（最快路径）
        # ================================================================
        print("[QR Pipeline] Step 1/3: zxing-cpp 全图扫描...")
        result = self._try_decode_zxing(img)
        if result:
            print(f"  [zxing] ✓ 全图识别成功！")
            print(f"  [zxing] 结果: {result[:100]}{'...' if len(result) > 100 else ''}")
            print("=" * 60 + "\n")
            self.last_crops = [img]
            return result
        print(f"  [zxing] ✗ 全图未能识别")

        # ================================================================
        # Step 2: OpenCV QRDetector 定位 → 裁剪 → WeChatQRCode
        # ================================================================
        print("[QR Pipeline] Step 2/3: OpenCV QRDetector 定位 → WeChatQRCode 识别...")
        qr_region, located, _ = self._locate_qr_opencv(img)

        if located and qr_region is not None:
            rh, rw = qr_region.shape[:2]
            print(f"[QR Pipeline] ✓ 定位成功，裁剪区域: {rw}x{rh}")
            self.last_crops = [qr_region]

            if self.qr_detector is not None:
                print(f"  [wx] 对定位区域进行多尺度扫描...")
                result = self._scan_multiscale_wx(qr_region)
                if result:
                    print(f"  [wx] ✓ 识别成功！")
                    print(f"  [wx] 结果: {result[:100]}{'...' if len(result) > 100 else ''}")
                    print("=" * 60 + "\n")
                    return result
                print(f"  [wx] ✗ 未能识别")
            else:
                print(f"  [wx] ⊘ 引擎未就绪")
            print("[QR Pipeline] 定位成功但 wx 未命中")
            print("=" * 60 + "\n")
            return None

        # ================================================================
        # Step 3: 兜底裁剪 → WeChatQRCode
        # ================================================================
        print("[QR Pipeline] Step 3/3: 兜底裁剪 → WeChatQRCode 扫描...")
        regions = self._generate_fallback_crops(img)
        self.last_crops = [r for _, r in regions]
        for i, (name, crop) in enumerate(regions):
            print(f"[QR Pipeline]   兜底区域[{i}] {name}: {crop.shape[1]}x{crop.shape[0]}")
        print(f"[QR Pipeline] 待扫描区域共 {len(regions)} 个")

        for idx, (region_name, region) in enumerate(regions):
            if region is None or region.size == 0:
                continue

            rh, rw = region.shape[:2]
            print(f"\n[QR Pipeline] ┌─ 区域 [{idx + 1}/{len(regions)}] {region_name} ({rw}x{rh}) ─┐")

            if self.qr_detector is not None:
                print(f"  [wx] 尝试 WeChatQRCode 多尺度扫描...")
                result = self._scan_multiscale_wx(region)
                if result:
                    print(f"  [wx] ✓ 识别成功！")
                    print(f"  [wx] 结果: {result[:100]}{'...' if len(result) > 100 else ''}")
                    return result
                else:
                    print(f"  [wx] ✗ 未能识别")
            else:
                print(f"  [wx] ⊘ 引擎未就绪，跳过")

            print(f"[QR Pipeline] └─ 区域 [{idx + 1}/{len(regions)}] 未命中 ─┘")

        print("\n[QR Pipeline] ✗ 所有区域、所有引擎均未能识别二维码")
        print("=" * 60 + "\n")
        return None
