import streamlit as st
import cv2
import re
from rapidocr_onnxruntime import RapidOCR
import json
import os
import sys
from PIL import Image
import numpy as np
import logging
import requests
import urllib3
import fitz  # PyMuPDF 用于处理 PDF 文件
from urllib.parse import unquote
import gc
import pandas as pd

# =========================================================
# 【核心模块】引入 DrissionPage，用于完全静默且防反爬的浏览器自动化操作
# =========================================================
from DrissionPage import ChromiumPage, ChromiumOptions
from DrissionPage.errors import ElementNotFoundError
import time

# 禁用 requests 在抓取 HTTPS 接口时产生的安全请求警告 (兼容政务内网环境)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================================================
# 核心业务逻辑类：负责图像增强、OCR 识别、二维码解析及信息结构化提取
# =========================================================
class ContractOCRService:
    def __init__(self):
        # 1. 初始化 RapidOCR 引擎 (使用 ONNX 运行时，轻量且在 CPU 上推理极快)
        try:
            self.ocr = RapidOCR()
        except Exception as e:
            st.error(f"RapidOCR 初始化失败: {e}")
            self.ocr = None

        # 2. 微信开源的超强二维码识别引擎配置 (比 OpenCV 原生的好用很多，支持倾斜和轻微模糊)
        self.model_dir = "wechat_models"
        self.wechat_models = {
            "detect.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.prototxt",
            "detect.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.caffemodel",
            "sr.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.prototxt",
            "sr.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.caffemodel"
        }
        # 加载引擎
        self.qr_detector = self._init_wechat_qrcode()

        # 3. 政务系统后端接口配置 (用于通过二维码解析出的 URL 获取不动产核心数据)
        self.target_api_url = "https://bdc.heyuan.gov.cn/actionapi/ZDTFHT/GetInfo"
        self.fixed_params = {
            "JGID": "FC830662-EA75-427C-9A82-443B91E383CB",
            "SJLY": "0q"
        }

    def _init_wechat_qrcode(self):
        """
        初始化微信二维码引擎。如果本地不存在模型文件，则自动从 GitHub 镜像下载。
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        paths = {}
        missing = False
        for filename, url in self.wechat_models.items():
            file_path = os.path.join(self.model_dir, filename)
            paths[filename] = file_path
            # 如果缺少文件则尝试网络下载（增加重试和异常捕捉防止崩溃）
            if not os.path.exists(file_path):
                missing = True
                try:
                    resp = requests.get(url, timeout=30, verify=False)
                    with open(file_path, 'wb') as f:
                        f.write(resp.content)
                except Exception:
                    pass

        try:
            return cv2.wechat_qrcode_WeChatQRCode(
                paths["detect.prototxt"], paths["detect.caffemodel"],
                paths["sr.prototxt"], paths["sr.caffemodel"]
            )
        except Exception:
            return None

    def _enhance_image(self, img):
        """
        图像增强逻辑：使用 CLAHE (对比度受限自适应直方图均衡化) 和高斯锐化，
        提升复印件、暗斑图像的文字对比度，显著提高 OCR 识别率。
        """
        if img is None: return None
        # 转灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 限制对比度的直方图均衡
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # 高斯模糊后，使用 addWeighted 进行图像锐化
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def _format_date_part(self, raw_str):
        """格式化日期片段，确保补零对齐（如把 '5' 转为 '05'）"""
        nums = re.findall(r'\d+', str(raw_str))
        if nums:
            val = int(nums[0])
            if val > 100: val = val % 100
            return f"{val:02d}"
        return "01"

    def _clean_noise(self, text):
        """清理 OCR 识别出的常见下划线、破折号及括号噪音"""
        if not text: return ""
        cleaned = re.sub(r'[_\-—~]+', ' ', text)
        cleaned = re.sub(r'^[（(\s]+', '', cleaned)
        cleaned = re.sub(r'[）(\s]+$', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
        
    def _find_qr_crops(self, img):
        """
        【已针对“附图页”进行算法优化】
        智能寻找疑似二维码区域并进行局部裁剪。
        通过边缘检测、形态学运算定位图像中疑似二维码的方形高密区域。
        """
        crops = []
        if img is None: return crops
        
        h, w = img.shape[:2]
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scale = 1.0
            # 图片过大时进行缩放，加速运算速度并过滤高频噪点
            if max(h, w) > 2000:
                scale = 1000.0 / max(h, w)
                gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
                
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Canny 边缘检测提取二维码密集特征
            edges = cv2.Canny(blurred, 50, 150)
            
            # 形态学闭运算：把二维码中密集的黑白点连接成一个“实心大方块”
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 【优化点】：增加形态学开运算
            # 作用：去除细长的干扰线（如虚线框）或孤立的文字块，只保留大面积实心区块
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
            
            # 寻找外轮廓
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            center_x, center_y = int(w * scale // 2), int(h * scale // 2)
            candidates = []
            
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                area = cw * ch
                img_area = (h * w * scale * scale)
                
                # 过滤条件1：面积过滤（排除全页或者微小噪点）
                if area < (img_area * 0.005) or area > (img_area * 0.4):
                    continue
                    
                # 过滤条件2：长宽比过滤（二维码是绝对的正方形）
                aspect_ratio = float(cw) / max(ch, 1)
                # 【优化点】：收紧长宽比阈值，有效过滤掉容易被误判的纵向虚线大边框
                if 0.8 < aspect_ratio < 1.2:
                    cx, cy = x + cw // 2, y + ch // 2
                    # 距离图像中心的距离（大多数带有二维码的附页，二维码均居中）
                    dist = (cx - center_x)**2 + (cy - center_y)**2
                    candidates.append((dist, x, y, cw, ch))
            
            # 按距离页面中心的距离升序排序，取最中间的候选框
            candidates.sort(key=lambda item: item[0])
            
            for _, x, y, cw, ch in candidates[:3]:
                # 还原至原始超大图分辨率的坐标
                real_x = int(x / scale)
                real_y = int(y / scale)
                real_cw = int(cw / scale)
                real_ch = int(ch / scale)
                
                # 【优化点】：增加自适应 Padding（静区保护）
                # 二维码识别必须要有足够的白边，否则引擎会失效
                pad = max(60, int(real_cw * 0.2))
                x1 = max(0, real_x - pad)
                y1 = max(0, real_y - pad)
                x2 = min(w, real_x + real_cw + pad)
                y2 = min(h, real_y + real_ch + pad)
                
                crops.append(img[y1:y2, x1:x2])
        except Exception as e:
            print(f"智能裁剪二维码区域异常: {e}")

        # ==================================
        # 兜底策略：如果图像极其复杂导致智能形态学失效，强行进行中央物理裁剪
        # ==================================
        # 兜底 1：中心 1/2 区域（常规合同文件）
        cx_crop = img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        crops.append(cx_crop)
        
        # 兜底 2：中心 2/3 区域（防止附图页周围留白极大，1/2刚好把二维码边缘给切断）
        cx_crop_large = img[h // 6 : 5 * h // 6, w // 6 : 5 * w // 6]
        crops.append(cx_crop_large)
        
        return crops

    def process_file(self, uploaded_file):
        """处理上传的文件流：区分 PDF 和普通图片，调用 OCR 并寻找二维码"""
        gc.collect() # 触发垃圾回收，防止内存泄漏
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name.lower()
        
        ocr_result = None
        qr_img = None
        preview_img = None 

        if file_name.endswith('.pdf'):
            doc = None
            try:
                # 使用 PyMuPDF 加载 PDF 流
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    if len(doc) > 0:
                        # ===== 处理首页：用于 OCR 文字提取 =====
                        page_1 = doc[0]
                        try:
                            # 2.0 矩阵渲染，保证 OCR 能看清字
                            mat = fitz.Matrix(2.0, 2.0) 
                            pix_1 = page_1.get_pixmap(matrix=mat)
                        except RuntimeError:
                            # 如果内存不足，降级渲染
                            print("[Warning] 高清渲染失败，尝试降级渲染...")
                            mat = fitz.Matrix(1.5, 1.5)
                            pix_1 = page_1.get_pixmap(matrix=mat)
                        
                        # 转换为 OpenCV 格式的 BGR 图像
                        img_data_1 = pix_1.samples
                        img_1 = Image.frombytes("RGB", [pix_1.width, pix_1.height], img_data_1)
                        cv_img_1 = cv2.cvtColor(np.array(img_1), cv2.COLOR_RGB2BGR)
                        del pix_1, img_data_1 
                        
                        preview_img = cv2.cvtColor(cv_img_1, cv2.COLOR_BGR2RGB) 
                        
                        # 图像增强后送入 RapidOCR
                        enhanced_img_1 = self._enhance_image(cv_img_1)
                        if self.ocr:
                            rapid_res, _ = self.ocr(enhanced_img_1)
                            if rapid_res:
                                # 格式转换，包裹为类似 PaddleOCR 的列表结构以复用原有提取逻辑
                                ocr_result = [[[box, (text, float(score))] for box, text, score in rapid_res]]
                    
                    if len(doc) > 0:
                        # ===== 处理末页：用于二维码识别（不动产合同二维码一般在最后一页附图页） =====
                        last_page_idx = len(doc) - 1
                        last_page = doc[last_page_idx]
                        pix_last = last_page.get_pixmap() 
                        
                        img_data_last = pix_last.samples
                        img_last = Image.frombytes("RGB", [pix_last.width, pix_last.height], img_data_last)
                        qr_img = cv2.cvtColor(np.array(img_last), cv2.COLOR_RGB2BGR)
                        del pix_last, img_data_last

            except Exception as e:
                st.error(f"PDF 处理异常: {e}")
                return None, None, None
            finally:
                gc.collect()
        else:
            # ===== 处理纯图片上传的情况 =====
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            preview_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            enhanced_img = self._enhance_image(img_bgr)
            
            if self.ocr:
                rapid_res, _ = self.ocr(enhanced_img)
                if rapid_res:
                    ocr_result = [[[box, (text, float(score))] for box, text, score in rapid_res]]
            qr_img = img_bgr 

        # 终端调试输出
        if ocr_result:
            print("\n" + "="*30 + " [DEBUG] 原始 OCR 识别结果 " + "="*30)
            if len(ocr_result) > 0 and ocr_result[0] is not None:
                for i, line in enumerate(ocr_result[0]):
                    text = line[1][0]
                    score = line[1][1]
                    print(f"[{i:02d}] {text}  (置信度: {score:.2f})")
            else:
                print("OCR 返回结果为空")
            print("="*80 + "\n")
        
        return ocr_result, qr_img, preview_img

    def extract_key_info(self, ocr_result):
        """
        结构化提取逻辑：基于正则匹配和关键字定位，
        从杂乱无章的 OCR 返回结果中抽离核心合同要素。
        """
        if not ocr_result or ocr_result[0] is None:
            return {
                "合同编号": "未找到", "抵押人": "未找到", "抵押人2": "",
                "证件号码": "未找到", "证件号码2": "", "债权数额": "未找到",
                "起始时间": "未找到", "结束时间": "未找到"
            }

        full_text_lines = [str(line[1][0]) for line in ocr_result[0]]
        full_text = "\n".join(full_text_lines)
        
        extracted = {
            "合同编号": "未找到", "抵押人": "未找到", "抵押人2": "",
            "证件号码": "未找到", "证件号码2": "", "债权数额": "未找到",
            "起始时间": "未找到", "结束时间": "未找到"
        }

        # 1. 提取合同编号：匹配“合同编号：”后面的连续字符
        m = re.search(r"合同编号[：:](.*?)(?:\n|$)", full_text)
        if m: 
            raw_no = m.group(1).strip()
            extracted["合同编号"] = re.sub(r'[_\-—]+', '', raw_no).strip()

        # 2. 提取抵押人：支持顿号或空格分隔提取出主/次抵押人
        mortgagor_pattern = r"抵押人\s*[：:]\s*([^（(\n]+)"
        mortgagors = re.findall(mortgagor_pattern, full_text)
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

        # 3. 提取证件号码：使用 18 位身份证严谨正则。如果包含字母X则转大写。
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
            clean_full_content = re.sub(r'[\s\-_—、，,]+', '', full_text)
            fallback_ids = re.findall(r'[1-9]\d{16}[0-9Xx]', clean_full_content)
            for i in fallback_ids:
                i_upper = i.upper()
                if i_upper not in ids:
                    ids.append(i_upper)
                    
        if len(ids) > 0:
            extracted["证件号码"] = ids[0]
        if len(ids) > 1:
            extracted["证件号码2"] = ids[1]

        # 4. 提取债权数额：过滤干扰符保留纯数字与小数点
        m = re.search(r"人民币\s*([\d\.\-_—]+)\s*(万?元)", full_text)
        if m:
            num_str = self._clean_noise(m.group(1)).replace(' ', '')
            clean_num = re.sub(r'[^\d\.]', '', num_str) 
            extracted["债权数额"] = clean_num
        else:
            m_alt = re.search(r"([\d\.]+)\s*万元", full_text)
            if m_alt:
                extracted["债权数额"] = m_alt.group(1)

        # 5. 提取履行时间：匹配格式“xxxx年xx月xx日至xxxx年xx月xx日”
        target_line = ""
        for i, text in enumerate(full_text_lines):
            if "履行期限" in text:
                target_line = text
                if i + 1 < len(full_text_lines):
                    target_line += " " + full_text_lines[i+1]
                break
        if not target_line: target_line = full_text
        norm_text = re.sub(r'(?<=\d日)\s*([至72ij/z~])\s*(?=\d{4}年)', '至', target_line)
        clean_term = re.sub(r'[_\-—~]+', '', norm_text)
        date_pattern = r"(\d{4})年(.*?)月(.*?)日"
        start_dt, end_dt = None, None

        if "至" in clean_term:
            parts = clean_term.split("至", 1)
            s_match = re.search(date_pattern, parts[0])
            if s_match:
                start_dt = f"{s_match.group(1)}-{self._format_date_part(s_match.group(2))}-{self._format_date_part(s_match.group(3))} 00:00:00"
            e_match = re.search(date_pattern, parts[1])
            if e_match:
                end_dt = f"{e_match.group(1)}-{self._format_date_part(e_match.group(2))}-{self._format_date_part(e_match.group(3))} 00:00:00"

        if not start_dt or not end_dt:
            all_matches = re.findall(date_pattern, clean_term)
            if len(all_matches) >= 2:
                if not start_dt:
                    start_dt = f"{all_matches[0][0]}-{self._format_date_part(all_matches[0][1])}-{self._format_date_part(all_matches[0][2])} 00:00:00"
                if not end_dt:
                    end_dt = f"{all_matches[-1][0]}-{self._format_date_part(all_matches[-1][1])}-{self._format_date_part(all_matches[-1][2])} 00:00:00"

        # 如果提取失败赋予默认值
        extracted["起始时间"] = start_dt if start_dt else f"{time.strftime('%Y-%m-%d', time.localtime())} 00:00:00"
        extracted["结束时间"] = end_dt if end_dt else "2036-12-31 00:00:00"
        return extracted

    def scan_and_fetch(self, img):
        """扫描二维码并在解析出 URL 后向后端发包拉取详细业务数据"""
        if self.qr_detector is None or img is None:
            return None, "引擎未就绪或图片为空"

        def _fetch_api(qr_text):
            try:
                # 切割URL携带的参数并构建 POST 表单
                params_str = qr_text.split("?")[-1]
                params = [unquote(p) for p in params_str.split('&')]
                payload = self.fixed_params.copy()
                keys = ["BDCDYID", "YWID", "BDCLX"]
                for i, key in enumerate(keys):
                    if i < len(params): payload[key] = params[i]

                headers = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}
                # 请求后端政务接口获取权属等机密信息
                r = requests.post(self.target_api_url, data=payload, headers=headers, timeout=10, verify=False)
                if r.status_code == 200:
                    return r.json(), "Success"
                else:
                    return None, f"HTTP {r.status_code}"
            except Exception as e:
                return None, str(e)

        # 按先后顺序遍历形态学切割出的局部区域进行识别
        crops = self._find_qr_crops(img)
        for crop in crops:
            if crop is not None and crop.size > 0:
                res, _ = self.qr_detector.detectAndDecode(crop)
                if res and res[0]:
                    print("[QR] 局部裁剪区域识别成功！")
                    return _fetch_api(res[0])
                    
        # 兜底全图识别
        print("[QR] 局部识别失效，尝试全图识别...")
        res, _ = self.qr_detector.detectAndDecode(img)
        if res and res[0]:
            return _fetch_api(res[0])

        return None, "未识别到二维码"


def read_json_config(config_path="config.json"):
    """读取本地的 JSON 配置文件，例如用于自动填充用户名和密码以及常用电话号码库"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = json.load(f)
            return config
        except json.JSONDecodeError:
            raise ValueError("配置文件格式错误（非合法JSON）")

# =========================================================
# 【DrissionPage 自动化流程区】
# 使用更为稳定、抗干扰且速度极快的 DrissionPage 接管浏览器
# =========================================================

def init_browser_and_visit_login():
    """初始化浏览器设置，并导航到登录页面"""
    co = ChromiumOptions()
    # 自动获取或启动浏览器，并防止代码结束后浏览器异常关闭
    co.auto_port() 
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chrome_binary_path = os.path.join(current_dir, "chrome_env", "chrome-win64", "chrome.exe")
    
    # 【智能适配】优先尝试内置便携版 Chrome；若无，则自动接管用户系统的浏览器
    if os.path.exists(chrome_binary_path):
        co.set_browser_path(chrome_binary_path)

    try:
        # 启动并接管浏览器页面
        page = ChromiumPage(co)
        page.set.window.max()
        
        login_url = "https://bdc.heyuan.gov.cn/yhbs/login"
        page.get(login_url)
        return page
    except Exception as e:
        st.error(f"浏览器启动失败！错误详情: {e}")
        return None

def login(page, username, password):
    """处理政务网特有的复选框登录逻辑"""
    try:
        # 输入账号（DP自带重试和等待机制，无需繁琐的 WebDriverWait）
        username_input = page.ele('xpath://input[@class="el-input__inner" and @placeholder="请输入姓名"]')
        username_input.input(username, clear=True)
        print("1：账号输入完成")
        
        # 输入密码
        password_input = page.ele('xpath://input[@class="el-input__inner" and @placeholder="请输入密码"]')
        password_input.input(password, clear=True)
        print("2：密码输入完成")

        # 勾选同意复选框
        agree_checkbox = page.ele('xpath://label[@class="el-checkbox"]//span[text()="同意"]/preceding-sibling::span[@class="el-checkbox__input"]')
        original_checkbox = agree_checkbox.ele('xpath:./input[@class="el-checkbox__original"]')
        
        if not original_checkbox.states.is_checked:
            agree_checkbox.click()
            print("3：同意复选框已勾选")
        else:
            print("3：同意复选框已勾选，无需重复操作")
        
        # 点击登录
        login_btn = page.ele('xpath://*[@id="app"]/div/div/div[1]/form/div/div[2]/div[5]/div/button')
        login_btn.click()
        print("4：登录按钮已点击")
        
        # 验证是否成功跳转至系统后台 (限定15秒)
        page.ele('xpath://*[@id="app"]/div/div/div[1]/div[1]/ul', timeout=15)
        print("登录成功！")
        return True
    
    except Exception as e:
        print(f"登录失败：{str(e)}")
        return False

def click_target_element(page, clickPATH):
    """基础辅助方法：点击目标 XPath"""
    try:
        target_element = page.ele(f'xpath:{clickPATH}', timeout=15)
        target_element.click()
        print("目标元素点击成功！")
        return True
    except Exception as e:
        print(f"点击目标元素失败：{str(e)}")
        return False

def web_input(input_data):      
    """
    【核心自动填单流水线】
    结合提取到的文本与接口数据，依次填写“抵押权人”、“抵押物”、“抵押信息”以及“抵押人”。
    """
    # ===== 会话持久化机制：复用页面，防止每填一条就重新登录 =====
    if 'browser_page' not in st.session_state:
        st.session_state.browser_page = None

    page = st.session_state.browser_page
    login_success = True

    if page is not None:
        try:
            # 探测当前浏览器页面是否存活（未被用户手动叉掉）
            _ = page.url
            page.refresh()
            print("✅ 探测到存活的浏览器会话，继续复用该页面进行多次录入...")
        except Exception:
            print("⚠️ 浏览器似乎已被手动关闭，准备重新启动新页面...")
            page = None

    if page is None:
        print("🚗 启动新的浏览器实例...")
        page = init_browser_and_visit_login()
        if page is None:
            return 
            
        st.session_state.browser_page = page
        try:
            config = read_json_config()
            USERNAME = config["bdc"]["username"]
            PASSWORD = config["bdc"]["password"]
            login_success = login(page, USERNAME, PASSWORD)
        except Exception as e:
            print(f"读取配置失败或登录失败：{e}")
    # =========================================================
    
    if login_success:
        # 导航并依次展开各级菜单
        click_success = click_target_element(page,'//*[@id="app"]/div/div/div[1]/div[1]/ul/div[2]/li/div')
        click_success = click_target_element(page,'//*[@id="app"]/div/div/div[1]/div[1]/ul/div[2]/li/ul/li/ul/div[1]/li/div')
        click_success = click_target_element(page,'//*[@id="app"]/div/div/div[1]/div[1]/ul/div[2]/li/ul/li/ul/div[1]/li/ul/li/ul/div[1]/li')

        # [阶段 1] 选择公共信息
        click_success = click_target_element(page,'//*[@id="tab-customTabs_17669806073123117"]')
        click_success = click_target_element(page,'//*[@id="pane-customTabs_17669806073123117"]/div/div/form/div[1]/div/div/div/input')
        click_success = click_target_element(page,'/html/body/div[3]/div[1]/div[1]/ul/li[3]')

        # [阶段 2] 抵押权人信息填入
        click_success = click_target_element(page,'//*[@id="tab-verticalCollapse_1766980689878640"]')
        click_success = click_target_element(page,'//*[@id="pane-verticalCollapse_1766980689878640"]/div/div/form/div[2]/div/div/div/input')
        click_success = click_target_element(page,'//li[@class="el-select-dropdown__item" and normalize-space(span/text())="中国工商银行股份有限公司龙川支行"]')
        
        try:
            cert_input = page.ele('xpath://*[@id="pane-verticalCollapse_1766980689878640"]/div/div/form/div[3]/div/div/input', timeout=10)
            time.sleep(0.5)
            cert_input.input(input_data['抵押权人联系电话'], clear=True)
        except Exception as e:
            print(f"输入抵押权人联系电话失败：{e}")

        # [阶段 3] 录入抵押物信息 (来自于二维码 API)
        click_success = click_target_element(page,'//*[@id="tab-verticalCollapse_1766980690997417"]')

        if click_success:
            input_success = False
            try:
                cert_input = page.ele('xpath://label[text()="不动产权证号"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10)
                cert_input.input(input_data['不动产证号'], clear=True)
                
                unit_input = page.ele('xpath://label[text()="不动产单元号"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10)
                unit_input.input(input_data['不动产单元号'], clear=True)
                input_success = True
            except Exception as e:
                print(f"❌ 不动产信息输入错误：{str(e)}")
            
            if input_success: print("不动产信息输入流程完成！")
        else:
            print("未执行后续操作：目标元素点击失败")

        # [阶段 4] 录入抵押具体信息 (来自于 OCR)
        click_success = click_target_element(page,'//*[@id="tab-verticalCollapse_176698069204667"]')
        if click_success:
            input_success = False
            try:
                # 动态选择抵押方式（依据合同编号的规则）
                click_target_element(page,"//label[text()='抵押方式']/following-sibling::div//input[@class='el-input__inner' and @readonly='readonly']")
                if '高额' in str(input_data['抵押合同号']):
                    click_target_element(page,'//*[text()="最高额抵押"]')
                else:
                    click_target_element(page,'//*[text()="一般抵押"]')

                # 其他要素逐一填列
                page.ele('xpath://label[text()="抵押顺位"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10).input(input_data['抵押顺位'], clear=True)
                page.ele('xpath://label[text()="抵押合同号"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10).input(input_data['抵押合同号'], clear=True)
                page.ele('xpath://label[text()="被担保主债权数额(万元)"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10).input(input_data['债权数额'], clear=True)
                page.ele('xpath://label[text()="债务履行起始时间"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10).input(input_data['起始时间'], clear=True)
                page.ele('xpath://label[text()="债务履行结束时间"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10).input(input_data['结束时间'], clear=True)
                page.ele('xpath://label[text()="担保范围"]/following-sibling::div//textarea[@class="el-textarea__inner"]', timeout=10).input(input_data['担保范围'], clear=True)
                
                input_success = True
            except Exception as e:
                print(f"❌ 抵押信息输入未知错误：{str(e)}")
            
            if input_success: print("抵押信息录入流程完成！")
        else:
            print("未执行后续操作：目标元素点击失败")

        print("\n🎉 系统主干字段录入完毕，准备录入抵押人...")
 
        # [阶段 5] 录入抵押人信息（包含新增行与 ElementUI 下拉框特殊处理）
        click_success = click_target_element(page,'//*[@id="tab-customTabs_17669806074173358"]')

        def fill_mortgagor_info(index, name, id_card, phone):
            print(f"--- 开始录入第 {index+1} 个抵押人信息: {name} ---")
            
            # --- 处理输入型字段：使用基于位置的批量抓取 + is_displayed 过滤 ---
            name_inputs = page.eles('xpath://label[contains(text(),"抵押人名称")]/following-sibling::div//input[@class="el-input__inner"]')
            visible_name_inputs = [ele for ele in name_inputs if ele.states.is_displayed]
            
            if len(visible_name_inputs) > index:
                visible_name_inputs[index].input(name, clear=True)
                print("已填写抵押人名称")
            else:
                print(f"❌ 未找到第 {index+1} 个可见的抵押人名称输入框")
                return 

            # --- 处理选择型字段：彻底解决幽灵下拉图层的点击拦截 ---
            type_inputs = page.eles('xpath://label[contains(text(),"抵押人证件类型")]/following-sibling::div//input')
            visible_type_inputs = [ele for ele in type_inputs if ele.states.is_displayed]
            
            if len(visible_type_inputs) > index:
                # 强行通过 JS 点击下拉框，防止动画遮盖导致的无法点击
                visible_type_inputs[index].click(by_js=True)
                print("已点击证件类型下拉框")
                time.sleep(0.8) 
                
                # ElementUI 会把渲染好的下拉菜单放在 <body> 最末尾
                dropdowns = page.eles('xpath://div[contains(@class, "el-select-dropdown")]')
                
                option_clicked = False
                # 倒序遍历页面中积累的所有下拉框组件，避开 display: none 的残留缓存层
                for dp in reversed(dropdowns):
                    style_str = dp.attr("style") or ""
                    if "display: none" not in style_str and "display:none" not in style_str:
                        target_opt = dp.ele('xpath:.//li[.//span[text()="身份证"]]')
                        if target_opt:
                            target_opt.click(by_js=True)
                            option_clicked = True
                            print("已选中：身份证")
                            break
                            
                if not option_clicked:
                    print("❌ 错误：下拉框已打开，但未找到当前可见的‘身份证’选项")
            else:
                print(f"❌ 未找到第 {index+1} 个可见的证件类型输入框")

            # 继续填入证件号码和电话
            id_inputs = page.eles('xpath://label[contains(text(),"抵押人证件号码")]/following-sibling::div//input[@class="el-input__inner"]')
            visible_id_inputs = [ele for ele in id_inputs if ele.states.is_displayed]
            
            if len(visible_id_inputs) > index:
                visible_id_inputs[index].input(id_card, clear=True)
                print("已填写证件号码")

            phone_inputs = page.eles('xpath://label[contains(text(),"抵押人联系电话")]/following-sibling::div//input[@class="el-input__inner"]')
            visible_phone_inputs = [ele for ele in phone_inputs if ele.states.is_displayed]
            
            if len(visible_phone_inputs) > index:
                visible_phone_inputs[index].input(phone, clear=True)
                print("已填写联系电话")
            
            print(f"=== 抵押人 {name} 录入完成 ===")

        # [执行] 循环录入抵押人
        if input_data.get('抵押人名称'):
            fill_mortgagor_info(0, input_data['抵押人名称'], input_data['抵押人证件号码'], input_data['抵押人联系电话'])
        
        if input_data.get('抵押人2名称'):
            try:
                # 存在两人以上时，点击新增按钮动态插入 DOM
                click_target_element(page, '//button[contains(@class, "el-button--success") and contains(@class, "el-button--mini") and span/text()="添加"]')
                print("已点击新增抵押人按钮")
                time.sleep(1.0) # 必须要等 Vue 组件渲染完毕
            except Exception as e:
                print("未找到新增按钮，如系统已有两行可忽略该错误：" + str(e))
            
            fill_mortgagor_info(1, input_data['抵押人2名称'], input_data['抵押人2证件号码'], input_data['抵押人联系电话'])
        
        # 任务终点：跳转到最后的提交大表检查页，交由人工二次复核并提交
        click_success = click_target_element(page,'//*[@id="tab-verticalCollapse_1766980692997735"]')
        print("✅ 自动化填单完成，请人工复核或手动关闭浏览器进行下一条录入！")


# =========================================================
# 【前端 UI 区】Streamlit 布局渲染
# =========================================================

st.set_page_config(
    page_title="河源不动产信息自动录入终端",
    page_icon="📄",
    layout="wide" # 使用全宽布局适配多列显示
)

def get_service():
    """工厂方法：实例化并在内存中保有核心工作组件"""
    return ContractOCRService()

def main():
    st.header("📄 河源不动产信息自动录入终端（V3.1 BY ZeroS）")
    
    # Session State 初始化，用于在 Streamlit 的重绘机制中长久保留计算好的数据
    if 'last_file_id' not in st.session_state: st.session_state.last_file_id = None
    if 'ocr_info' not in st.session_state: st.session_state.ocr_info = None
    if 'api_data' not in st.session_state: st.session_state.api_data = None
    if 'preview_img' not in st.session_state: st.session_state.preview_img = None

    uploaded_file = st.file_uploader("请上传合同文件 (PDF 或 图片)", type=["pdf", "jpg", "png", "jpeg"])

    if uploaded_file:
        service = get_service()
        
        # 使用文件名+体积作为文件唯一标识，防止组件重渲染引发的无限重复 OCR
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        f_key = file_id 
        
        if st.session_state.last_file_id != file_id:
            st.session_state.preview_img = None 
            st.session_state.ocr_info = None
            st.session_state.api_data = None
            gc.collect()

            with st.spinner('正在进行智能识别与政务网数据交互，请稍候...'):
                ocr_result, qr_img_cv, preview_img = service.process_file(uploaded_file)
                
                st.session_state.last_file_id = file_id
                st.session_state.preview_img = preview_img
                
                if ocr_result and len(ocr_result) > 0 and ocr_result[0] is not None:
                    st.session_state.ocr_info = service.extract_key_info(ocr_result)
                else:
                    st.session_state.ocr_info = None
                
                if qr_img_cv is not None:
                    api_data, msg = service.scan_and_fetch(qr_img_cv)
                    if api_data and str(api_data.get("Code")) == "0":
                        st.session_state.api_data = api_data.get("Data", [])[0] if api_data.get("Data") else None
                    else:
                        st.session_state.api_data = None
                else:
                    st.session_state.api_data = None

        st.markdown("---")
        
        # 建立网格试图，分为图片预览、OCR 提取结果修正区、API结果联动区三块
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📑 合同首页预览")
            if st.session_state.preview_img is not None:
                st.image(st.session_state.preview_img, use_container_width=True, caption="首页 OCR 扫描区域")
            else:
                st.warning("无法生成预览图")

        with col2:
            st.subheader("📝 文本提取结果")
            info = st.session_state.ocr_info
            
            if info:
                def update_contract_no():
                    sel = st.session_state.get(f"template_sel_{f_key}")
                    if sel and sel != "使用原始提取值":
                        st.session_state[f"no_{f_key}"] = sel

                templates = [
                    "使用原始提取值",
                    "02006000292026二手贷7000",
                    "02006000292026一手贷7000",
                    "02006000292026消费经营组合贷7000、02006000292026高额贷（抵）7000"
                ]
                st.selectbox(
                    "快捷合同模板选择", 
                    templates, 
                    key=f"template_sel_{f_key}", 
                    on_change=update_contract_no
                )
                
                # 双向绑定设计，用户更改输入框内容后，值保留在 session 对应的 key 里
                contract_no = st.text_input("合同编号", value=info['合同编号'], key=f"no_{f_key}")
                mortgagor = st.text_input("抵押人", value=info['抵押人'], key=f"mort_{f_key}")
                mortgagor2 = st.text_input("抵押人2", value=info.get('抵押人2', ''), key=f"mort2_{f_key}")
                id_card = st.text_input("证件号码", value=info['证件号码'], key=f"id_{f_key}")
                id_card2 = st.text_input("证件号码2", value=info.get('证件号码2', ''), key=f"id2_{f_key}")
                amount = st.text_input("债权数额", value=info['债权数额'], key=f"amt_{f_key}")
                
                d1, d2 = st.columns(2)
                with d1:
                    start_date = st.text_input("起始时间", value=info['起始时间'], key=f"start_{f_key}")
                with d2:
                    end_date = st.text_input("结束时间", value=info['结束时间'], key=f"end_{f_key}")
            else:
                st.error("OCR 识别失败，未能提取到有效文字。")
                contract_no, mortgagor, mortgagor2, id_card, id_card2, amount, start_date, end_date = "", "", "", "", "", "", "", ""

        with col3:
            st.subheader("📱 实时接口数据")
            d = st.session_state.api_data
            
            if d:
                # 只读展示接口拉取来的防伪特征项
                zh = st.text_input("权证号 (ZH)", value=d.get('ZH', '无'), disabled=True, key=f"api_zh_{f_key}")
                zl = st.text_input("房屋坐落 (ZL)", value=d.get('ZL', '无'), disabled=True, key=f"api_zl_{f_key}")
                jzmj = st.text_input("建筑面积 (JZMJ)", value=d.get('JZMJ', '无'), disabled=True, key=f"api_mj_{f_key}")
                bdcdyh = st.text_input("不动产单元号 (BDCDYH)", value=d.get('BDCDYH', '无'), disabled=True, key=f"api_bdc_{f_key}")
            else:
                st.warning("接口未返回有效数据，请手动输入权证号和单元号")
                zh = st.text_input("权证号 (ZH)", value="", key=f"api_zh_{f_key}")
                zl = st.text_input("房屋坐落 (ZL)", value="", key=f"api_zl_{f_key}")
                jzmj = st.text_input("建筑面积 (JZMJ)", value="", key=f"api_mj_{f_key}")
                bdcdyh = st.text_input("不动产单元号 (BDCDYH)", value="", key=f"api_bdc_{f_key}")

        st.markdown("---")
        
        # 收集经用户复核过后的终态数据字典
        current_contract_no = st.session_state.get(f"no_{f_key}", "")
        current_mortgagor = st.session_state.get(f"mort_{f_key}", "")
        current_mortgagor2 = st.session_state.get(f"mort2_{f_key}", "")
        current_id_card = st.session_state.get(f"id_{f_key}", "")
        current_id_card2 = st.session_state.get(f"id2_{f_key}", "")
        current_amount = st.session_state.get(f"amt_{f_key}", "")
        current_start_date = st.session_state.get(f"start_{f_key}", "")
        current_end_date = st.session_state.get(f"end_{f_key}", "")
        current_zh = st.session_state.get(f"api_zh_{f_key}", "")
        current_bdcdyh = st.session_state.get(f"api_bdc_{f_key}", "")
        
        # 尝试由外挂 JSON 文件热加载电话簿
        phone_options = [{"name": "请检查配置文件", "phone": "NULL"}]
        try:
            config_data = read_json_config()
            if "phones" in config_data and isinstance(config_data["phones"], list) and len(config_data["phones"]) > 0:
                valid_phones = []
                for item in config_data["phones"]:
                    if isinstance(item, dict) and item.get("phone") and item.get("name"):
                        valid_phones.append({
                            "name": str(item["name"]).strip(),
                            "phone": str(item["phone"]).strip()
                        })
                if valid_phones:
                    phone_options = valid_phones
        except Exception:
            pass

        phone_display = [f"{item['name']} {item['phone']}" for item in phone_options]
        phone_values = [item["phone"] for item in phone_options]

        _, _, btn_col = st.columns([2, 2, 1])
        with col3:
            # 构建用户手机号选择器
            selected_idx = st.selectbox(
                "选择抵押权人联系电话",
                range(len(phone_display)),
                format_func=lambda x: phone_display[x],
                key=f"phone_{f_key}"
            )
            selected_phone = phone_values[selected_idx]
            
            # 生成投递到 DrissionPage 的序列化字典
            input_data_flat = {
                "抵押人名称": current_mortgagor,
                "抵押人2名称": current_mortgagor2,
                "抵押人联系电话": "6753094",
                "抵押权人联系电话": selected_phone,
                "抵押人证件号码": current_id_card,
                "抵押人2证件号码": current_id_card2,
                "不动产证号": current_zh,
                "不动产单元号": current_bdcdyh,
                "抵押方式": "",
                "抵押顺位": "1",
                "抵押合同号": current_contract_no,
                "债权数额": current_amount,
                "起始时间": current_start_date,
                "结束时间": current_end_date,
                "担保范围": "主债权本金、利息、罚息、复利、违约金、损害赔偿金以及实现抵押权的费用（包括但不限于诉讼费、律师费等）"
            }

            # 执行流水的最终触发按钮
            if st.button("录入系统"):
                with st.spinner("正在将数据推送到业务系统浏览器，请勿关闭..."):
                    web_input(input_data_flat)
                    st.success("浏览器操作结束，请在网页中人工复核或手动关闭。")

if __name__ == "__main__":
    main()