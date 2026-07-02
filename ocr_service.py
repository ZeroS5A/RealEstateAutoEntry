"""
OCR 文字识别与信息提取服务
=======================
负责图像预处理、RapidOCR 文字识别、以及不动产合同结构化信息提取。
"""

import re
import time
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


class OCRService:
    """OCR 识别 + 合同信息结构化提取"""

    def __init__(self):
        try:
            self.engine = RapidOCR()
        except Exception as e:
            print(f"[OCR] RapidOCR 初始化失败: {e}")
            self.engine = None

    # ==================================================================
    # 图像预处理
    # ==================================================================

    @staticmethod
    def enhance_image(img):
        """
        CLAHE 对比度增强 + 高斯锐化。
        提升复印件、暗斑图像的文字对比度，显著提高 OCR 识别率。
        """
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # ==================================================================
    # OCR 识别
    # ==================================================================

    def recognize(self, img_bgr):
        """
        对 BGR 图像执行 OCR 识别。
        返回格式化的 OCR 结果列表；引擎未就绪或无文字时返回 None。
        """
        if self.engine is None or img_bgr is None:
            return None
        enhanced = self.enhance_image(img_bgr)
        rapid_res, _ = self.engine(enhanced)
        if rapid_res:
            return [[[box, (text, float(score))] for box, text, score in rapid_res]]
        return None

    # ==================================================================
    # 文本辅助
    # ==================================================================

    @staticmethod
    def _format_date_part(raw_str):
        """格式化日期片段，确保补零对齐（如 '5' → '05'）"""
        nums = re.findall(r'\d+', str(raw_str))
        if nums:
            val = int(nums[0])
            if val > 100:
                val = val % 100
            return f"{val:02d}"
        return "01"

    @staticmethod
    def _clean_noise(text):
        """清理 OCR 常见噪音：下划线、破折号、括号"""
        if not text:
            return ""
        cleaned = re.sub(r'[_\-—~]+', ' ', text)
        cleaned = re.sub(r'^[（(\s]+', '', cleaned)
        cleaned = re.sub(r'[）(\s]+$', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    # ==================================================================
    # 结构化提取
    # ==================================================================

    def extract_key_info(self, ocr_result):
        """
        从 OCR 结果中结构化提取不动产合同核心要素：
        合同编号、抵押人、证件号码、债权数额、履行期限。
        """
        if not ocr_result or ocr_result[0] is None:
            return {
                "合同编号": "未找到", "抵押人": "未找到", "抵押人2": "",
                "证件号码": "", "证件号码2": "", "债权数额": "未找到",
                "起始时间": "未找到", "结束时间": "未找到"
            }

        full_text_lines = [str(line[1][0]) for line in ocr_result[0]]
        full_text = "\n".join(full_text_lines)

        extracted = {
            "合同编号": "未找到", "抵押人": "未找到", "抵押人2": "",
            "证件号码": "", "证件号码2": "", "债权数额": "未找到",
            "起始时间": "未找到", "结束时间": "未找到"
        }

        # 1. 合同编号
        m = re.search(r"合同编号[：:](.*?)(?:\n|$)", full_text)
        if m:
            raw_no = m.group(1).strip()
            extracted["合同编号"] = re.sub(r'[_\-—]+', '', raw_no).strip()

        # 2. 抵押人（支持顿号/空格分隔）
        mortgagors = re.findall(r"抵押人\s*[：:]\s*([^（(\n]+)", full_text)
        cleaned_mortgagors = []
        for mtg in mortgagors:
            clean_m = self._clean_noise(mtg)
            parts = re.split(r'[、,，\s]+', clean_m)
            for part in parts:
                part = part.strip()
                if part and part not in cleaned_mortgagors and part not in ["等", "及"]:
                    cleaned_mortgagors.append(part)

        if len(cleaned_mortgagors) > 0:
            extracted["抵押人"] = cleaned_mortgagors[0]
        if len(cleaned_mortgagors) > 1:
            extracted["抵押人2"] = cleaned_mortgagors[1]

        # 3. 证件号码（18 位身份证）
        ids = []
        for line in full_text_lines:
            if "证件号码" in line or "身份证" in line or "号码" in line:
                clean_line = re.sub(r'[\s\-_—、，,]+', '', line)
                line_ids = re.findall(r'[1-9]\d{16}[0-9Xx]', clean_line)
                for i in line_ids:
                    i_upper = i.upper()
                    if i_upper not in ids:
                        ids.append(i_upper)

        if not ids:
            clean_full = re.sub(r'[\s\-_—、，,]+', '', full_text)
            fallback_ids = re.findall(r'[1-9]\d{16}[0-9Xx]', clean_full)
            for i in fallback_ids:
                i_upper = i.upper()
                if i_upper not in ids:
                    ids.append(i_upper)

        if len(ids) > 0:
            extracted["证件号码"] = ids[0]
        if len(ids) > 1:
            extracted["证件号码2"] = ids[1]

        # 4. 债权数额
        m = re.search(r"人民币\s*([\d\.\-_—]+)\s*(万?元)", full_text)
        if m:
            num_str = self._clean_noise(m.group(1)).replace(' ', '')
            clean_num = re.sub(r'[^\d\.]', '', num_str)
            extracted["债权数额"] = clean_num
        else:
            m_alt = re.search(r"([\d\.]+)\s*万元", full_text)
            if m_alt:
                extracted["债权数额"] = m_alt.group(1)

        # 5. 履行期限
        target_line = ""
        for i, text in enumerate(full_text_lines):
            if "履行期限" in text:
                target_line = text
                if i + 1 < len(full_text_lines):
                    target_line += " " + full_text_lines[i + 1]
                break
        if not target_line:
            target_line = full_text

        norm_text = re.sub(r'(?<=\d日)\s*([至72ij/z~])\s*(?=\d{4}年)', '至', target_line)
        clean_term = re.sub(r'[_\-—~]+', '', norm_text)
        date_pattern = r"(\d{4})年(.*?)月(.*?)日"
        start_dt, end_dt = None, None

        if "至" in clean_term:
            parts = clean_term.split("至", 1)
            s_match = re.search(date_pattern, parts[0])
            if s_match:
                start_dt = (f"{s_match.group(1)}-"
                            f"{self._format_date_part(s_match.group(2))}-"
                            f"{self._format_date_part(s_match.group(3))} 00:00:00")
            e_match = re.search(date_pattern, parts[1])
            if e_match:
                end_dt = (f"{e_match.group(1)}-"
                          f"{self._format_date_part(e_match.group(2))}-"
                          f"{self._format_date_part(e_match.group(3))} 00:00:00")

        if not start_dt or not end_dt:
            all_matches = re.findall(date_pattern, clean_term)
            if len(all_matches) >= 2:
                if not start_dt:
                    start_dt = (f"{all_matches[0][0]}-"
                                f"{self._format_date_part(all_matches[0][1])}-"
                                f"{self._format_date_part(all_matches[0][2])} 00:00:00")
                if not end_dt:
                    end_dt = (f"{all_matches[-1][0]}-"
                              f"{self._format_date_part(all_matches[-1][1])}-"
                              f"{self._format_date_part(all_matches[-1][2])} 00:00:00")

        extracted["起始时间"] = start_dt if start_dt else f"{time.strftime('%Y-%m-%d', time.localtime())} 00:00:00"
        extracted["结束时间"] = end_dt if end_dt else ""
        return extracted
