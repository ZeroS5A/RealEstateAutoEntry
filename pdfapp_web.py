import streamlit as st
import cv2
import re
from paddleocr import PaddleOCR
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
import gc  # 引入垃圾回收模块
import pandas as pd # 引入pandas用于处理CSV

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementNotInteractableException
from selenium.webdriver.common.keys import Keys # 【优化点】补充漏掉的Keys导入，防止高阶输入助手报错
import time

# --- 新增：国内镜像与稳定性配置 ---
# 强制指定 webdriver_manager 使用国内镜像源 (npmmirror)
# os.environ['WDM_CHROME_METADATA_URL'] = "https://registry.npmmirror.com/-/binary/chromedriver"
# os.environ['WDM_SOURCE'] = "https://registry.npmmirror.com/-/binary/chromedriver"
# 禁用 SSL 验证，防止因公司内网证书拦截导致的连接失败
os.environ['WDM_SSL_VERIFY'] = '0'

# 禁用安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------
# 核心逻辑类 (基于 V1.2 修改以适配 Web 流)
# ---------------------------------------------------------

class ContractOCRService:
    def __init__(self):
        # 1. 初始化 PaddleOCR
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True, 
                lang="ch",
                enable_mkldnn=False,
                use_gpu=False,
                show_log=False
            )
        except TypeError:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

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
        """初始化微信二维码引擎，Web版增加自动下载提示"""
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
        """图像增强逻辑 V1.2"""
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

    def process_file(self, uploaded_file):
        """处理上传的文件流"""
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
                            print("[Warning] 高清渲染失败，尝试降级渲染...")
                            mat = fitz.Matrix(1.5, 1.5)
                            pix_1 = page_1.get_pixmap(matrix=mat)
                        
                        img_data_1 = pix_1.samples
                        img_1 = Image.frombytes("RGB", [pix_1.width, pix_1.height], img_data_1)
                        cv_img_1 = cv2.cvtColor(np.array(img_1), cv2.COLOR_RGB2BGR)
                        del pix_1, img_data_1 
                        
                        preview_img = cv2.cvtColor(cv_img_1, cv2.COLOR_BGR2RGB) 
                        
                        enhanced_img_1 = self._enhance_image(cv_img_1)
                        ocr_result = self.ocr.ocr(enhanced_img_1)
                    
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
            ocr_result = self.ocr.ocr(enhanced_img)
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
        """提取逻辑"""
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

        # 1. 合同编号
        m = re.search(r"合同编号[：:](.*?)(?:\n|$)", full_text)
        if m: 
            raw_no = m.group(1).strip()
            extracted["合同编号"] = re.sub(r'[_\-—]+', '', raw_no).strip()

        # 2. 抵押人 (支持提取多个，优化顿号、空格及下划线分隔)
        # 修改正则，去掉\s限制，允许匹配到顿号、逗号、空格等，直到遇到换行或左括号
        mortgagor_pattern = r"抵押人\s*[：:]\s*([^（(\n]+)"
        mortgagors = re.findall(mortgagor_pattern, full_text)
        cleaned_mortgagors = []
        for mtg in mortgagors:
            clean_m = self._clean_noise(mtg)
            # 针对多个抵押人用顿号、逗号、空格分隔的情况进行拆分
            parts = re.split(r'[、,，\s]+', clean_m)
            for part in parts:
                part = part.strip()
                # 过滤空值及常见无意义的字词
                if part and part not in cleaned_mortgagors and part not in ["等", "及"]:
                    cleaned_mortgagors.append(part)
                
        if len(cleaned_mortgagors) > 0:
            extracted["抵押人"] = cleaned_mortgagors[0]
        if len(cleaned_mortgagors) > 1:
            extracted["抵押人2"] = cleaned_mortgagors[1]

        # 3. 证件号码 (支持提取多个，优化下划线、顿号干扰)
        ids = []
        for line in full_text_lines:
            # 增加"号码"等宽泛关键字，防止OCR少识别字
            if "证件号码" in line or "身份证" in line or "号码" in line:
                # 将可能的下划线、破折号、空格、顿号、逗号等干扰符全部清除
                clean_line = re.sub(r'[\s\-_—、，,]+', '', line)
                # 提取纯净的18位身份证号
                line_ids = re.findall(r'[1-9]\d{16}[0-9Xx]', clean_line)
                for i in line_ids:
                    i_upper = i.upper() # 统一转换为大写X
                    if i_upper not in ids:
                        ids.append(i_upper)
        
        # fallback: 如果含有关键字的行没有提取到，则全文扫描一次
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

        # 4. 债权数额 (只保留数字和小数点)
        m = re.search(r"人民币\s*([\d\.\-_—]+)\s*(万?元)", full_text)
        if m:
            num_str = self._clean_noise(m.group(1)).replace(' ', '')
            clean_num = re.sub(r'[^\d\.]', '', num_str) # 剔除杂项，只保留数字和小数点
            extracted["债权数额"] = clean_num
        else:
            # 容错：不带“人民币”前缀的情况
            m_alt = re.search(r"([\d\.]+)\s*万元", full_text)
            if m_alt:
                extracted["债权数额"] = m_alt.group(1)

        # 5. 履行时间
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
        """扫描二维码并请求数据"""
        if self.qr_detector is None or img is None:
            return None, "引擎未就绪或图片为空"

        res, _ = self.qr_detector.detectAndDecode(img)
        if not res or not res[0]:
            return None, "未识别到二维码"

        try:
            params_str = res[0].split("?")[-1]
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

def read_json_config(config_path="config.json"):
    """
    读取.json格式的配置文件
    :param config_path: 配置文件路径
    :return: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    
    # 读取并解析JSON文件
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = json.load(f)
            return config
        except json.JSONDecodeError:
            raise ValueError("配置文件格式错误（非合法JSON）")

# -------------------------- 1. 初始化浏览器，访问登录页 --------------------------
def init_browser_and_visit_login():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    # 开启 detach 参数。执行结束浏览器依旧保持打开状态
    chrome_options.add_experimental_option("detach", True)
    
    # --- 离线方案核心路径配置 ---
    # 获取当前 py 文件所在的目录 (ContractOCR_Portable)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 拼接便携版 Chrome.exe 的绝对路径
    chrome_binary_path = os.path.join(current_dir, "chrome_env", "chrome-win64", "chrome.exe")
    # 拼接 ChromeDriver.exe 的绝对路径
    driver_path = os.path.join(current_dir, "chrome_env", "chromedriver-win64", "chromedriver.exe")
    
    # 指定我们要使用的免安装版浏览器路径
    chrome_options.binary_location = chrome_binary_path

    try:
        # 直接通过 Service 加载本地驱动，彻底断绝网络依赖
        service = Service(executable_path=driver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        st.error("离线浏览器启动失败！请检查路径是否正确配置。")
        st.code(f"预期的浏览器路径: {chrome_binary_path}\n预期的驱动路径: {driver_path}\n报错详情: {e}")
        return None
        
    login_url = "https://bdc.heyuan.gov.cn/yhbs/login"
    try:
        driver.get(login_url)
        driver.maximize_window()
        return driver
    except Exception as e:
        st.error(f"无法访问登录地址（请检查这台电脑能否连接政务网）: {e}")
        return None


# -------------------------- 2. 执行登录操作（核心） --------------------------
def login(driver, username, password):
    try:
        wait = WebDriverWait(driver, 10)
        
        username_input = wait.until(EC.element_to_be_clickable((By.XPATH, '//input[@class="el-input__inner" and @placeholder="请输入姓名"]')))
        username_input.clear()
        username_input.send_keys(username)
        print("1：账号输入完成")
        
        password_input = wait.until(EC.element_to_be_clickable((By.XPATH, '//input[@class="el-input__inner" and @placeholder="请输入密码"]')))
        password_input.clear()
        password_input.send_keys(password)
        print("2：密码输入完成")

        agree_checkbox = wait.until(EC.element_to_be_clickable((
            By.XPATH,
            '//label[@class="el-checkbox"]//span[text()="同意"]/preceding-sibling::span[@class="el-checkbox__input"]'
        )))
        
        original_checkbox = agree_checkbox.find_element(By.XPATH, './input[@class="el-checkbox__original"]')
        if not original_checkbox.is_selected():
            agree_checkbox.click()
            print("3：同意复选框已勾选")
        else:
            print("3：同意复选框已勾选，无需重复操作")
        
        login_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="app"]/div/div/div[1]/form/div/div[2]/div[5]/div/button')))
        login_btn.click()
        print("4：登录按钮已点击")
        
        wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="app"]/div/div/div[1]/div[1]/ul'))) 
        print("登录成功！")
        return True
    
    except Exception as e:
        print(f"登录失败：{str(e)}")
        driver.save_screenshot("login_error.png")
        return False

# -------------------------- 3. 点击指定XPath元素 --------------------------
def click_target_element(driver,clickPATH):
    try:
        wait = WebDriverWait(driver, 15)
        target_element = wait.until(
            EC.element_to_be_clickable((By.XPATH, clickPATH))
        )
        target_element.click()
        print("目标元素点击成功！")
        return True
    except Exception as e:
        print(f"点击目标元素失败：{str(e)}")
        driver.save_screenshot("click_element_error.png")
        return False

def web_input(input_data):      
    # ===== 【优化点】使用 st.session_state 缓存 driver 对象，实现浏览器实例和页面状态复用 =====
    if 'browser_driver' not in st.session_state:
        st.session_state.browser_driver = None

    driver = st.session_state.browser_driver
    login_success = True  # 假设复用时已处于登录状态

    if driver is not None:
        try:
            # 探测当前浏览器是否还存活（是否被用户手动叉掉）
            _ = driver.current_url
            driver.refresh()
            print("✅ 探测到存活的浏览器会话，继续复用该页面进行多次录入...")
        except Exception:
            print("⚠️ 浏览器似乎已被手动关闭，准备重新启动新页面...")
            driver = None

    if driver is None:
        # 只有在没有缓存，或者浏览器被手动关闭时，才启动新的实例并登录
        print("🚗 启动新的浏览器实例...")
        driver = init_browser_and_visit_login()
        if driver is None:
            return # 如果驱动初始化失败，直接返回，避免程序崩溃
            
        st.session_state.browser_driver = driver
        try:
            config = read_json_config()
            USERNAME = config["bdc"]["username"]
            PASSWORD = config["bdc"]["password"]
            login_success = login(driver, USERNAME, PASSWORD)
        except Exception as e:
            print(f"读取配置失败：{e}")
    # =========================================================================================
    
    if login_success:
        click_success = click_target_element(driver,'//*[@id="app"]/div/div/div[1]/div[1]/ul/div[2]/li/div')
        click_success = click_target_element(driver,'//*[@id="app"]/div/div/div[1]/div[1]/ul/div[2]/li/ul/li/ul/div[1]/li/div')
        click_success = click_target_element(driver,'//*[@id="app"]/div/div/div[1]/div[1]/ul/div[2]/li/ul/li/ul/div[1]/li/ul/li/ul/div[1]/li')

        # 选择公共信息
        click_success = click_target_element(driver,'//*[@id="tab-customTabs_17669806073123117"]')
        click_success = click_target_element(driver,'//*[@id="pane-customTabs_17669806073123117"]/div/div/form/div[1]/div/div/div/input')
        click_success = click_target_element(driver,'/html/body/div[3]/div[1]/div[1]/ul/li[3]')

        # 选择抵押权人信息
        click_success = click_target_element(driver,'//*[@id="tab-verticalCollapse_1766980689878640"]')
        click_success = click_target_element(driver,'//*[@id="pane-verticalCollapse_1766980689878640"]/div/div/form/div[2]/div/div/div/input')
        click_success = click_target_element(driver,'//li[@class="el-select-dropdown__item" and normalize-space(span/text())="中国工商银行股份有限公司龙川支行"]')
        cert_input = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="pane-verticalCollapse_1766980689878640"]/div/div/form/div[3]/div/div/input'))
        )
        cert_input.clear()
        cert_input.send_keys(input_data['抵押权人联系电话'])

        # 录入抵押物信息
        click_success = click_target_element(driver,'//*[@id="tab-verticalCollapse_1766980690997417"]')

        if click_success:
            input_success = False  # 修复局部变量未初始化的问题
            try:
                cert_input = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//label[text()="不动产权证号"]/following-sibling::div//input[@class="el-input__inner"]'))
                )
                cert_input.send_keys(input_data['不动产证号'])
                print(f"已输入不动产证号：{input_data['不动产证号']}")
                
                unit_input = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//label[text()="不动产单元号"]/following-sibling::div//input[@class="el-input__inner"]'))
                )
                unit_input.send_keys(input_data['不动产单元号'])
                print(f"已输入不动产单元号：{input_data['不动产单元号']}")
                
                input_success = True
            except TimeoutException:
                print("❌ 超时错误：未在10秒内找到目标输入框，请检查XPath或页面加载状态")
            except ElementNotInteractableException:
                print("❌ 交互错误：输入框不可输入（可能被遮挡/禁用）")
            except Exception as e:
                print(f"❌ 未知错误：{str(e)}")
            
            if input_success:
                print("不动产信息输入流程完成！")
            else:
                print("不动产信息输入失败！")
            # time.sleep(5)
        else:
            print("未执行后续操作：目标元素点击失败")

        # 录入抵押信息
        click_success = click_target_element(driver,'//*[@id="tab-verticalCollapse_176698069204667"]')
        if click_success:
            input_success = False  # 修复局部变量未初始化的问题
            try:
                click_success = click_target_element(driver,"//label[text()='抵押方式']/following-sibling::div//input[@class='el-input__inner' and @readonly='readonly']")
                if '高额' in str(input_data['抵押合同号']):
                    click_success = click_target_element(driver,'//*[text()="最高额抵押"]')
                else:
                    click_success = click_target_element(driver,'//*[text()="一般抵押"]')

                cert_input = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//label[text()="抵押顺位"]/following-sibling::div//input[@class="el-input__inner"]'))
                )
                cert_input.send_keys(input_data['抵押顺位'])
                
                unit_input = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//label[text()="抵押合同号"]/following-sibling::div//input[@class="el-input__inner"]'))
                )
                unit_input.send_keys(input_data['抵押合同号'])

                unit_input = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//label[text()="被担保主债权数额(万元)"]/following-sibling::div//input[@class="el-input__inner"]'))
                )
                unit_input.send_keys(input_data['债权数额'])
                unit_input = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//label[text()="债务履行起始时间"]/following-sibling::div//input[@class="el-input__inner"]'))
                )
                unit_input.send_keys(input_data['起始时间'])
                unit_input = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//label[text()="债务履行结束时间"]/following-sibling::div//input[@class="el-input__inner"]'))
                )
                unit_input.send_keys(input_data['结束时间'])

                

                unit_input = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//label[text()="担保范围"]/following-sibling::div//textarea[@class="el-textarea__inner"]'))
                )
                unit_input.send_keys(input_data['担保范围'])
                
                input_success = True
            except TimeoutException:
                print("❌ 超时错误：未在10秒内找到目标输入框，请检查XPath或页面加载状态")
            except ElementNotInteractableException:
                print("❌ 交互错误：输入框不可输入（可能被遮挡/禁用）")
            except Exception as e:
                print(f"❌ 未知错误：{str(e)}")
            
            if input_success:
                print("不动产信息输入流程完成！")
            else:
                print("不动产信息输入失败！")
            # time.sleep(5)
        else:
            print("未执行后续操作：目标元素点击失败")

        print("\n🎉 系统所有关联字段录入操作执行完毕！等待人工复核...")
        # # =========================================================

 
        # 录入抵押人信息
        click_success = click_target_element(driver,'//*[@id="tab-customTabs_17669806074173358"]')

        # ===== 新增：录入抵押人信息逻辑 =====
        def fill_mortgagor_info(index, name, id_card, phone):
            print(f"--- 开始录入第 {index+1} 个抵押人信息: {name} ---")
            
            # ================= 1. 抵押人名称 =================
            # 获取所有“抵押人名称”后的输入框
            name_xpath = '//label[contains(text(),"抵押人名称")]/following-sibling::div//input[@class="el-input__inner"]'
            name_inputs = driver.find_elements(By.XPATH, name_xpath)
            
            # 筛选出当前页面所有【可见】的输入框
            visible_name_inputs = [ele for ele in name_inputs if ele.is_displayed()]
            
            if len(visible_name_inputs) > index:
                visible_name_inputs[index].clear()
                visible_name_inputs[index].send_keys(name)
                print("已填写抵押人名称")
            else:
                print(f"❌ 未找到第 {index+1} 个可见的抵押人名称输入框")
                return # 找不到后续就没法填了，直接返回

            # ================= 2. 抵押人证件类型 -> 身份证 =================
            # 这一步是报错的高发区，改用最稳健的“可见性判断”法
            
            # 2.1 定位输入框：找到所有可见的“证件类型”输入框
            type_input_xpath = '//label[contains(text(),"抵押人证件类型")]/following-sibling::div//input'
            type_inputs = driver.find_elements(By.XPATH, type_input_xpath)
            visible_type_inputs = [ele for ele in type_inputs if ele.is_displayed()]
            
            if len(visible_type_inputs) > index:
                # 点击输入框，触发下拉菜单
                driver.execute_script("arguments[0].click();", visible_type_inputs[index])
                print("已点击证件类型下拉框")
                
                # 2.2 强制等待下拉菜单动画完成 (Element UI 必须步骤)
                # time.sleep(1) 
                
                # 2.3 在所有“身份证”选项中，点击那个【可见】的
                # 注意：这里查找的是全局的 dropdown item，不局限于某个 id
                options = driver.find_elements(By.XPATH, "//li[contains(@class, 'el-select-dropdown__item') and .//span[text()='身份证']]")
                
                option_clicked = False
                for opt in options:
                    if opt.is_displayed():
                        opt.click()
                        option_clicked = True
                        print("已选中：身份证")
                        break
                
                if not option_clicked:
                    print("❌ 错误：下拉框已打开，但未找到可见的‘身份证’选项")
            else:
                print(f"❌ 未找到第 {index+1} 个可见的证件类型输入框")

            # ================= 3. 抵押人证件号码 =================
            id_xpath = '//label[contains(text(),"抵押人证件号码")]/following-sibling::div//input[@class="el-input__inner"]'
            id_inputs = driver.find_elements(By.XPATH, id_xpath)
            visible_id_inputs = [ele for ele in id_inputs if ele.is_displayed()]
            
            if len(visible_id_inputs) > index:
                visible_id_inputs[index].clear()
                visible_id_inputs[index].send_keys(id_card)
                print("已填写证件号码")

            # ================= 4. 抵押人联系电话 =================
            phone_xpath = '//label[contains(text(),"抵押人联系电话")]/following-sibling::div//input[@class="el-input__inner"]'
            phone_inputs = driver.find_elements(By.XPATH, phone_xpath)
            visible_phone_inputs = [ele for ele in phone_inputs if ele.is_displayed()]
            
            if len(visible_phone_inputs) > index:
                visible_phone_inputs[index].clear()
                visible_phone_inputs[index].send_keys(phone)
                print("已填写联系电话")
            
            print(f"=== 抵押人 {name} 录入完成 ===")

        # 录入抵押人 1
        if input_data.get('抵押人名称'):
            fill_mortgagor_info(0, input_data['抵押人名称'], input_data['抵押人证件号码'], input_data['抵押人联系电话'])
        
        # 录入抵押人 2 (如果有)
        if input_data.get('抵押人2名称'):
            try:
                # 尝试点击新增按钮
                click_target_element(driver,'//button[contains(@class, "el-button--success") and contains(@class, "el-button--mini") and span/text()="添加"]')
                print("已点击新增抵押人按钮")
                time.sleep(1.5) # 等待DOM渲染新表单行
            except Exception as e:
                print("未找到或无法点击新增按钮，如系统已有两行可忽略该错误：" + str(e))
            
            fill_mortgagor_info(1, input_data['抵押人2名称'], input_data['抵押人2证件号码'], input_data['抵押人联系电话'])
        # ==================================
        
        # 跳转到业务提交页
        click_success = click_target_element(driver,'//*[@id="tab-verticalCollapse_1766980692997735"]')

        # 【优化点】删除了 time.sleep(600)。由于开启了 detach，执行结束后浏览器将由人工管理并保持开启，这不仅能节省内存卡顿也让页面瞬间解除正在执行的状态。
        print("✅ 自动化填单完成，请人工核职并手动关闭浏览器或进行下一条录入！")


# ---------------------------------------------------------
# Streamlit 界面逻辑
# ---------------------------------------------------------

st.set_page_config(
    page_title="河源不动产信息自动录入终端",
    page_icon="📄",
    layout="wide"
)

def get_service():
    return ContractOCRService()

def main():
    st.header("📄 河源不动产信息自动录入终端（V2.2 BY ZeroS）")
    
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

                # 合并后的模板选项
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
                
                # 抵押人1及2
                mortgagor = st.text_input("抵押人", value=info['抵押人'], key=f"mort_{f_key}")
                mortgagor2 = st.text_input("抵押人2", value=info.get('抵押人2', ''), key=f"mort2_{f_key}")
                
                # 身份证1及2
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
        
        # 无论 API 是否有数据，统一从 session_state 获取，确保兼容手动输入的值
        current_zh = st.session_state.get(f"api_zh_{f_key}", "")
        current_bdcdyh = st.session_state.get(f"api_bdc_{f_key}", "")
        
        # === 新增：读取手机号码列表配置 ===
        phone_list = ["6753094"] # 默认保底值
        try:
            config_data = read_json_config()
            # 检查是否有配置phones数组
            if "phones" in config_data and isinstance(config_data["phones"], list) and len(config_data["phones"]) > 0:
                phone_list = [str(p) for p in config_data["phones"]] # 确保读取出来的是字符串
        except Exception:
            pass
        # ==================================

        # 【修改重点 1、2】修复了with语法，将直接调用函数改为了通过 Streamlit 按钮判定后再执行
        _, _, btn_col = st.columns([2, 2, 1])
        with col3:
            # --- 新增下拉框在按钮上方 ---
            selected_phone = st.selectbox("选择抵押权人联系电话", phone_list, key=f"phone_{f_key}")
            
            # 【修改重点 3】这里改为了普通字典对象，不使用列表包裹，防止selenium send_keys报错
            input_data_flat = {
                "抵押人名称": current_mortgagor,
                "抵押人2名称": current_mortgagor2,
                "抵押人联系电话": "6753094",
                "抵押权人联系电话": selected_phone,  # <--- 使用用户在下拉框中选中的手机号
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