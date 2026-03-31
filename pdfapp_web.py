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
import fitz  # PyMuPDF
from urllib.parse import unquote
import gc
import pandas as pd

# 【核心修改点：弃用 Selenium，引入 DrissionPage】
from DrissionPage import ChromiumPage, ChromiumOptions
from DrissionPage.errors import ElementNotFoundError
import time

# 禁用安全请求警告 (兼容公司内网环境)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------
# 核心逻辑类 (基于 V1.2 修改以适配 Web 流)
# ---------------------------------------------------------

class ContractOCRService:
    def __init__(self):
        # 1. 初始化 RapidOCR
        try:
            self.ocr = RapidOCR()
        except Exception as e:
            st.error(f"RapidOCR 初始化失败: {e}")
            self.ocr = None

        # 2. 微信二维码引擎配置
        self.model_dir = "wechat_models"
        self.wechat_models = {
            "detect.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.prototxt",
            "detect.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/detect.caffemodel",
            "sr.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.prototxt",
            "sr.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/wechat_qrcode/sr.caffemodel"
        }
        self.qr_detector = self._init_wechat_qrcode()

        # 3. API 配置
        self.target_api_url = "https://bdc.heyuan.gov.cn/actionapi/ZDTFHT/GetInfo"
        self.fixed_params = {
            "JGID": "FC830662-EA75-427C-9A82-443B91E383CB",
            "SJLY": "0q"
        }

    def _init_wechat_qrcode(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        paths = {}
        missing = False
        for filename, url in self.wechat_models.items():
            file_path = os.path.join(self.model_dir, filename)
            paths[filename] = file_path
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
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def _format_date_part(self, raw_str):
        nums = re.findall(r'\d+', str(raw_str))
        if nums:
            val = int(nums[0])
            if val > 100: val = val % 100
            return f"{val:02d}"
        return "01"

    def _clean_noise(self, text):
        if not text: return ""
        cleaned = re.sub(r'[_\-—~]+', ' ', text)
        cleaned = re.sub(r'^[（(\s]+', '', cleaned)
        cleaned = re.sub(r'[）(\s]+$', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
        
    def _find_qr_crops(self, img):
        crops = []
        if img is None: return crops
        
        h, w = img.shape[:2]
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scale = 1.0
            if max(h, w) > 2000:
                scale = 1000.0 / max(h, w)
                gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
                
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            center_x, center_y = int(w * scale // 2), int(h * scale // 2)
            candidates = []
            
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                area = cw * ch
                img_area = (h * w * scale * scale)
                
                if area < (img_area * 0.005) or area > (img_area * 0.4):
                    continue
                    
                aspect_ratio = float(cw) / max(ch, 1)
                if 0.7 < aspect_ratio < 1.3:
                    cx, cy = x + cw // 2, y + ch // 2
                    dist = (cx - center_x)**2 + (cy - center_y)**2
                    candidates.append((dist, x, y, cw, ch))
            
            candidates.sort(key=lambda item: item[0])
            
            for _, x, y, cw, ch in candidates[:3]:
                real_x = int(x / scale)
                real_y = int(y / scale)
                real_cw = int(cw / scale)
                real_ch = int(ch / scale)
                
                pad = max(50, int(real_cw * 0.2))
                x1 = max(0, real_x - pad)
                y1 = max(0, real_y - pad)
                x2 = min(w, real_x + real_cw + pad)
                y2 = min(h, real_y + real_ch + pad)
                
                crops.append(img[y1:y2, x1:x2])
        except Exception as e:
            print(f"智能裁剪二维码区域异常: {e}")

        cx_crop = img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        crops.append(cx_crop)
        
        return crops

    def process_file(self, uploaded_file):
        gc.collect()
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name.lower()
        
        ocr_result = None
        qr_img = None
        preview_img = None 

        if file_name.endswith('.pdf'):
            doc = None
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    if len(doc) > 0:
                        page_1 = doc[0]
                        try:
                            mat = fitz.Matrix(2.0, 2.0) 
                            pix_1 = page_1.get_pixmap(matrix=mat)
                        except RuntimeError:
                            mat = fitz.Matrix(1.5, 1.5)
                            pix_1 = page_1.get_pixmap(matrix=mat)
                        
                        img_data_1 = pix_1.samples
                        img_1 = Image.frombytes("RGB", [pix_1.width, pix_1.height], img_data_1)
                        cv_img_1 = cv2.cvtColor(np.array(img_1), cv2.COLOR_RGB2BGR)
                        del pix_1, img_data_1 
                        
                        preview_img = cv2.cvtColor(cv_img_1, cv2.COLOR_BGR2RGB) 
                        
                        enhanced_img_1 = self._enhance_image(cv_img_1)
                        if self.ocr:
                            rapid_res, _ = self.ocr(enhanced_img_1)
                            if rapid_res:
                                ocr_result = [[[box, (text, float(score))] for box, text, score in rapid_res]]
                    
                    if len(doc) > 0:
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
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            preview_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            enhanced_img = self._enhance_image(img_bgr)
            if self.ocr:
                rapid_res, _ = self.ocr(enhanced_img)
                if rapid_res:
                    ocr_result = [[[box, (text, float(score))] for box, text, score in rapid_res]]
            qr_img = img_bgr 

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
        if not ocr_result or ocr_result[0] is None:
            return {
                "合同编号": "未找到",
                "抵押人": "未找到",
                "抵押人2": "",
                "证件号码": "未找到",
                "证件号码2": "",
                "债权数额": "未找到",
                "起始时间": "未找到",
                "结束时间": "未找到"
            }

        full_text_lines = [str(line[1][0]) for line in ocr_result[0]]
        full_text = "\n".join(full_text_lines)
        
        extracted = {
            "合同编号": "未找到",
            "抵押人": "未找到",
            "抵押人2": "",
            "证件号码": "未找到",
            "证件号码2": "",
            "债权数额": "未找到",
            "起始时间": "未找到",
            "结束时间": "未找到"
        }

        m = re.search(r"合同编号[：:](.*?)(?:\n|$)", full_text)
        if m: 
            raw_no = m.group(1).strip()
            extracted["合同编号"] = re.sub(r'[_\-—]+', '', raw_no).strip()

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

        m = re.search(r"人民币\s*([\d\.\-_—]+)\s*(万?元)", full_text)
        if m:
            num_str = self._clean_noise(m.group(1)).replace(' ', '')
            clean_num = re.sub(r'[^\d\.]', '', num_str) 
            extracted["债权数额"] = clean_num
        else:
            m_alt = re.search(r"([\d\.]+)\s*万元", full_text)
            if m_alt:
                extracted["债权数额"] = m_alt.group(1)

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

        extracted["起始时间"] = start_dt if start_dt else f"{time.strftime('%Y-%m-%d', time.localtime())} 00:00:00"
        extracted["结束时间"] = end_dt if end_dt else "2036-12-31 00:00:00"
        return extracted

    def scan_and_fetch(self, img):
        if self.qr_detector is None or img is None:
            return None, "引擎未就绪或图片为空"

        def _fetch_api(qr_text):
            try:
                params_str = qr_text.split("?")[-1]
                params = [unquote(p) for p in params_str.split('&')]
                payload = self.fixed_params.copy()
                keys = ["BDCDYID", "YWID", "BDCLX"]
                for i, key in enumerate(keys):
                    if i < len(params): payload[key] = params[i]

                headers = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}
                r = requests.post(self.target_api_url, data=payload, headers=headers, timeout=10, verify=False)
                if r.status_code == 200:
                    return r.json(), "Success"
                else:
                    return None, f"HTTP {r.status_code}"
            except Exception as e:
                return None, str(e)

        crops = self._find_qr_crops(img)
        for crop in crops:
            if crop is not None and crop.size > 0:
                res, _ = self.qr_detector.detectAndDecode(crop)
                if res and res[0]:
                    print("[QR] 局部裁剪区域识别成功！")
                    return _fetch_api(res[0])
                    
        print("[QR] 局部识别失效，尝试全图识别...")
        res, _ = self.qr_detector.detectAndDecode(img)
        if res and res[0]:
            return _fetch_api(res[0])

        return None, "未识别到二维码"


def read_json_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = json.load(f)
            return config
        except json.JSONDecodeError:
            raise ValueError("配置文件格式错误（非合法JSON）")

# -------------------------- 1. 初始化浏览器，访问登录页 (DrissionPage 重构) --------------------------
def init_browser_and_visit_login():
    co = ChromiumOptions()
    # 自动获取或启动浏览器，并防止代码结束后浏览器关闭
    co.auto_port() 
    
    # 获取当前 py 文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chrome_binary_path = os.path.join(current_dir, "chrome_env", "chrome-win64", "chrome.exe")
    
    # 【智能适配】如果存在打包的便携版 Chrome 就用便携版；如果删除了，则自动接管系统原生 Edge/Chrome
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

# -------------------------- 2. 执行登录操作 (DrissionPage 重构) --------------------------
def login(page, username, password):
    try:
        # DP 具有自动等待机制，输入账号
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
        
        # 检查是否未勾选
        if not original_checkbox.states.is_checked:
            agree_checkbox.click()
            print("3：同意复选框已勾选")
        else:
            print("3：同意复选框已勾选，无需重复操作")
        
        # 点击登录
        login_btn = page.ele('xpath://*[@id="app"]/div/div/div[1]/form/div/div[2]/div[5]/div/button')
        login_btn.click()
        print("4：登录按钮已点击")
        
        # 等待后台主界面元素加载，证明登录成功 (最多等待 15 秒)
        page.ele('xpath://*[@id="app"]/div/div/div[1]/div[1]/ul', timeout=15)
        print("登录成功！")
        return True
    
    except Exception as e:
        print(f"登录失败：{str(e)}")
        # page.get_screenshot(path="login_error.png")
        return False

# -------------------------- 3. 点击指定XPath元素 (DrissionPage 重构) --------------------------
def click_target_element(page, clickPATH):
    try:
        target_element = page.ele(f'xpath:{clickPATH}', timeout=15)
        target_element.click()
        print("目标元素点击成功！")
        return True
    except Exception as e:
        print(f"点击目标元素失败：{str(e)}")
        return False

def web_input(input_data):      
    # ===== 使用 st.session_state 缓存 page 对象 =====
    if 'browser_page' not in st.session_state:
        st.session_state.browser_page = None

    page = st.session_state.browser_page
    login_success = True

    if page is not None:
        try:
            # 探测当前浏览器页面是否存活
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
    # =========================================================================================
    
    if login_success:
        click_success = click_target_element(page,'//*[@id="app"]/div/div/div[1]/div[1]/ul/div[2]/li/div')
        click_success = click_target_element(page,'//*[@id="app"]/div/div/div[1]/div[1]/ul/div[2]/li/ul/li/ul/div[1]/li/div')
        click_success = click_target_element(page,'//*[@id="app"]/div/div/div[1]/div[1]/ul/div[2]/li/ul/li/ul/div[1]/li/ul/li/ul/div[1]/li')

        # 选择公共信息
        click_success = click_target_element(page,'//*[@id="tab-customTabs_17669806073123117"]')
        click_success = click_target_element(page,'//*[@id="pane-customTabs_17669806073123117"]/div/div/form/div[1]/div/div/div/input')
        click_success = click_target_element(page,'/html/body/div[3]/div[1]/div[1]/ul/li[3]')

        # 选择抵押权人信息
        click_success = click_target_element(page,'//*[@id="tab-verticalCollapse_1766980689878640"]')
        click_success = click_target_element(page,'//*[@id="pane-verticalCollapse_1766980689878640"]/div/div/form/div[2]/div/div/div/input')
        click_success = click_target_element(page,'//li[@class="el-select-dropdown__item" and normalize-space(span/text())="中国工商银行股份有限公司龙川支行"]')
        
        try:
            cert_input = page.ele('xpath://*[@id="pane-verticalCollapse_1766980689878640"]/div/div/form/div[3]/div/div/input', timeout=10)
            time.sleep(0.5)
            cert_input.input(input_data['抵押权人联系电话'], clear=True)
        except Exception as e:
            print(f"输入抵押权人联系电话失败：{e}")

        # 录入抵押物信息
        click_success = click_target_element(page,'//*[@id="tab-verticalCollapse_1766980690997417"]')

        if click_success:
            input_success = False
            try:
                cert_input = page.ele('xpath://label[text()="不动产权证号"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10)
                cert_input.input(input_data['不动产证号'], clear=True)
                print(f"已输入不动产证号：{input_data['不动产证号']}")
                
                unit_input = page.ele('xpath://label[text()="不动产单元号"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10)
                unit_input.input(input_data['不动产单元号'], clear=True)
                print(f"已输入不动产单元号：{input_data['不动产单元号']}")
                
                input_success = True
            except Exception as e:
                print(f"❌ 不动产信息输入错误：{str(e)}")
            
            if input_success:
                print("不动产信息输入流程完成！")
            else:
                print("不动产信息输入失败！")
        else:
            print("未执行后续操作：目标元素点击失败")

        # 录入抵押信息
        click_success = click_target_element(page,'//*[@id="tab-verticalCollapse_176698069204667"]')
        if click_success:
            input_success = False
            try:
                # 抵押方式
                click_target_element(page,"//label[text()='抵押方式']/following-sibling::div//input[@class='el-input__inner' and @readonly='readonly']")
                if '高额' in str(input_data['抵押合同号']):
                    click_target_element(page,'//*[text()="最高额抵押"]')
                else:
                    click_target_element(page,'//*[text()="一般抵押"]')

                page.ele('xpath://label[text()="抵押顺位"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10).input(input_data['抵押顺位'], clear=True)
                page.ele('xpath://label[text()="抵押合同号"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10).input(input_data['抵押合同号'], clear=True)
                page.ele('xpath://label[text()="被担保主债权数额(万元)"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10).input(input_data['债权数额'], clear=True)
                page.ele('xpath://label[text()="债务履行起始时间"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10).input(input_data['起始时间'], clear=True)
                page.ele('xpath://label[text()="债务履行结束时间"]/following-sibling::div//input[@class="el-input__inner"]', timeout=10).input(input_data['结束时间'], clear=True)
                page.ele('xpath://label[text()="担保范围"]/following-sibling::div//textarea[@class="el-textarea__inner"]', timeout=10).input(input_data['担保范围'], clear=True)
                
                input_success = True
            except Exception as e:
                print(f"❌ 抵押信息输入未知错误：{str(e)}")
            
            if input_success:
                print("抵押信息录入流程完成！")
            else:
                print("抵押信息录入失败！")
        else:
            print("未执行后续操作：目标元素点击失败")

        print("\n🎉 系统主干字段录入完毕，准备录入抵押人...")
 
        # 录入抵押人信息
        click_success = click_target_element(page,'//*[@id="tab-customTabs_17669806074173358"]')

        # ===== DP版：录入抵押人信息逻辑 =====
        def fill_mortgagor_info(index, name, id_card, phone):
            print(f"--- 开始录入第 {index+1} 个抵押人信息: {name} ---")
            
            # ================= 1. 抵押人名称 =================
            name_inputs = page.eles('xpath://label[contains(text(),"抵押人名称")]/following-sibling::div//input[@class="el-input__inner"]')
            visible_name_inputs = [ele for ele in name_inputs if ele.states.is_displayed]
            
            if len(visible_name_inputs) > index:
                visible_name_inputs[index].input(name, clear=True)
                print("已填写抵押人名称")
            else:
                print(f"❌ 未找到第 {index+1} 个可见的抵押人名称输入框")
                return 

            # ================= 2. 抵押人证件类型 -> 身份证 =================
            type_inputs = page.eles('xpath://label[contains(text(),"抵押人证件类型")]/following-sibling::div//input')
            visible_type_inputs = [ele for ele in type_inputs if ele.states.is_displayed]
            
            if len(visible_type_inputs) > index:
                # 【改动1】使用 JS 强行触发点击，防止因遮挡导致的 Click Intercepted
                visible_type_inputs[index].click(by_js=True)
                print("已点击证件类型下拉框")
                time.sleep(0.8) # 给一点点下拉动画渲染时间
                
                # 【改动2：彻底解决 ElementUI 幽灵下拉框问题】
                # 获取页面里所有的下拉框容器
                dropdowns = page.eles('xpath://div[contains(@class, "el-select-dropdown")]')
                
                option_clicked = False
                # 倒序遍历：ElementUI 会将最新弹出的组件追加到 HTML body 最末尾
                for dp in reversed(dropdowns):
                    style_str = dp.attr("style") or ""
                    # 摒弃脆弱的 XPath 字符串匹配，使用更稳健的 Python 语法来过滤隐藏图层
                    if "display: none" not in style_str and "display:none" not in style_str:
                        # 在这个真正可见的下拉框中，精准定位“身份证”选项
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

            # ================= 3. 抵押人证件号码 =================
            id_inputs = page.eles('xpath://label[contains(text(),"抵押人证件号码")]/following-sibling::div//input[@class="el-input__inner"]')
            visible_id_inputs = [ele for ele in id_inputs if ele.states.is_displayed]
            
            if len(visible_id_inputs) > index:
                visible_id_inputs[index].input(id_card, clear=True)
                print("已填写证件号码")

            # ================= 4. 抵押人联系电话 =================
            phone_inputs = page.eles('xpath://label[contains(text(),"抵押人联系电话")]/following-sibling::div//input[@class="el-input__inner"]')
            visible_phone_inputs = [ele for ele in phone_inputs if ele.states.is_displayed]
            
            if len(visible_phone_inputs) > index:
                visible_phone_inputs[index].input(phone, clear=True)
                print("已填写联系电话")
            
            print(f"=== 抵押人 {name} 录入完成 ===")

        # 录入抵押人 1
        if input_data.get('抵押人名称'):
            fill_mortgagor_info(0, input_data['抵押人名称'], input_data['抵押人证件号码'], input_data['抵押人联系电话'])
        
        # 录入抵押人 2 (如果有)
        if input_data.get('抵押人2名称'):
            try:
                click_target_element(page, '//button[contains(@class, "el-button--success") and contains(@class, "el-button--mini") and span/text()="添加"]')
                print("已点击新增抵押人按钮")
                time.sleep(1.0) # 等待DOM渲染新表单行
            except Exception as e:
                print("未找到新增按钮，如系统已有两行可忽略该错误：" + str(e))
            
            fill_mortgagor_info(1, input_data['抵押人2名称'], input_data['抵押人2证件号码'], input_data['抵押人联系电话'])
        
        # 跳转到业务提交页
        click_success = click_target_element(page,'//*[@id="tab-verticalCollapse_1766980692997735"]')
        print("✅ 自动化填单完成，请人工复核或手动关闭浏览器进行下一条录入！")


# ---------------------------------------------------------
# Streamlit 界面逻辑 (完全无更改)
# ---------------------------------------------------------

st.set_page_config(
    page_title="河源不动产信息自动录入终端",
    page_icon="📄",
    layout="wide"
)

def get_service():
    return ContractOCRService()

def main():
    st.header("📄 河源不动产信息自动录入终端（V3.0 BY ZeroS）")
    
    if 'last_file_id' not in st.session_state:
        st.session_state.last_file_id = None
    if 'ocr_info' not in st.session_state:
        st.session_state.ocr_info = None
    if 'api_data' not in st.session_state:
        st.session_state.api_data = None
    if 'preview_img' not in st.session_state:
        st.session_state.preview_img = None

    uploaded_file = st.file_uploader("请上传合同文件 (PDF 或 图片)", type=["pdf", "jpg", "png", "jpeg"])

    if uploaded_file:
        service = get_service()
        
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        f_key = file_id 
        
        if st.session_state.last_file_id != file_id:
            st.session_state.preview_img = None 
            st.session_state.ocr_info = None
            st.session_state.api_data = None
            gc.collect()

            with st.spinner('正在进行智能识别，请稍候...'):
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
            selected_idx = st.selectbox(
                "选择抵押权人联系电话",
                range(len(phone_display)),
                format_func=lambda x: phone_display[x],
                key=f"phone_{f_key}"
            )
            selected_phone = phone_values[selected_idx]
            
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

            if st.button("录入系统"):
                with st.spinner("正在将数据推送到业务系统浏览器，请勿关闭..."):
                    web_input(input_data_flat)
                    st.success("浏览器操作结束，请在网页中人工复核或手动关闭。")

if __name__ == "__main__":
    main()