import openai
import base64

# Initialize OpenAI client


def encode_image(image_path):
    """Encodes an image to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(image_path):
    """Sends the image to OpenAI API for analysis"""
    image_data = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4-turbo",

        messages=[
            {"role": "system", "content": "You are an AI assistant that analyzes images."},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]}
        ],
        max_tokens=200
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    image_path = "C:/Users/shake/PycharmProjects/PythonProject/canvas_screenshots/full_page.png"  # Change this to your image path
    result = analyze_image(image_path)
    print("Image Analysis Result:\n", result)



    def capture_canvas_screenshots(self, url):
        """Capture screenshots of all canvas and image elements from a webpage."""

        self.clean_directory()

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920x1080")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, "canvas")))
            time.sleep(2)

            # Capture all canvas elements
            canvas_data = driver.execute_script("""
                   let canvases = document.querySelectorAll('canvas');
                   let dataUrls = [];
                   canvases.forEach((canvas, index) => {
                       let dataUrl = canvas.toDataURL("image/jpeg");
                       dataUrls.push({id: canvas.id || `canvas_${index+1}`, dataUrl: dataUrl});
                   });
                   return dataUrls;
               """)

            if not canvas_data:
                print("No canvas elements found on the page.")
            else:
                print(f"Found {len(canvas_data)} canvas elements. Extracting images...")

                for canvas in canvas_data:
                    canvas_id = canvas["id"]
                    data_url = canvas["dataUrl"]

                    # Remove the header "data:image/jpeg;base64,"
                    image_data = base64.b64decode(data_url.split(",")[1])

                    # Save the image as JPEG
                    img = Image.open(BytesIO(image_data))
                    img = img.convert("RGB")  # Ensure it's in RGB mode for JPEG
                    img.save(os.path.join(self.canvas_output_dir, f"{canvas_id}.jpg"), "JPEG")

                    print(f"Saved: {canvas_id}.jpg")

            # Capture all image elements
            img_elements = driver.find_elements(By.TAG_NAME, "img")

            if not img_elements:
                print("No image elements found on the page.")
            else:
                print(f"Found {len(img_elements)} image elements. Extracting images...")

                for index, img_element in enumerate(img_elements, start=1):
                    img_src = img_element.get_attribute("src")
                    if img_src.startswith("data:image"):
                        # Base64 encoded image
                        img_format = img_src.split("/")[1].split(";")[0]
                        img_data = base64.b64decode(img_src.split(",")[1])
                    else:
                        # URL-based image
                        img_data = requests.get(img_src).content

                    img = Image.open(BytesIO(img_data))
                    img = img.convert("RGB")  # Ensure it's in RGB mode for JPEG
                    img.save(os.path.join(self.canvas_output_dir, f"image_{index}.jpg"), "JPEG")

                    print(f"Saved: image_{index}.jpg")

        except Exception as e:
            print(f"Error capturing canvas and images: {e}")
            traceback.print_exc()

        finally:
            driver.quit()

            def capture_canvas_screenshots(self, url):
                """Capture screenshots of all canvas and image elements from a webpage."""

                self.clean_directory()

                options = Options()
                options.add_argument("--headless")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920x1080")

                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=options)

                try:
                    driver.get(url)
                    time.sleep(3)  # wait for page to start rendering

                    # Debug: save a screenshot right after loading
                    driver.save_screenshot("page_loaded_debug.png")
                    print("Saved initial page screenshot as page_loaded_debug.png")

                    # Optional: check if iframe exists
                    iframes = driver.find_elements(By.TAG_NAME, "iframe")
                    if iframes:
                        print(f"Found {len(iframes)} iframe(s). Switching to the first one.")
                        driver.switch_to.frame(iframes[0])

                    # Custom retry loop to wait for canvas
                    max_retries = 10
                    canvas_found = False
                    for i in range(max_retries):
                        canvas_elements = driver.find_elements(By.TAG_NAME, "canvas")
                        if canvas_elements:
                            print(f"Canvas found after {i + 1} seconds.")
                            canvas_found = True
                            break
                        print(f"Waiting for canvas... ({i + 1}s)")
                        time.sleep(1)

                    if not canvas_found:
                        print("No canvas elements found after waiting.")
                        driver.save_screenshot("no_canvas_debug.png")
                        return

                    # Capture all canvas elements
                    canvas_data = driver.execute_script("""
                        let canvases = document.querySelectorAll('canvas');
                        let dataUrls = [];
                        canvases.forEach((canvas, index) => {
                            try {
                                let dataUrl = canvas.toDataURL("image/jpeg");
                                dataUrls.push({id: canvas.id || `canvas_${index + 1}`, dataUrl: dataUrl});
                            } catch (err) {
                                console.error("Canvas toDataURL failed:", err);
                            }
                        });
                        return dataUrls;
                    """)

                    if not canvas_data:
                        print("No canvas data extracted.")
                    else:
                        print(f"Found {len(canvas_data)} canvas elements. Extracting images...")

                        for canvas in canvas_data:
                            canvas_id = canvas["id"]
                            data_url = canvas["dataUrl"]

                            # Remove the header "data:image/jpeg;base64,"
                            image_data = base64.b64decode(data_url.split(",")[1])

                            img = Image.open(BytesIO(image_data))
                            img = img.convert("RGB")
                            img.save(os.path.join(self.canvas_output_dir, f"{canvas_id}.jpg"), "JPEG")

                            print(f"Saved: {canvas_id}.jpg")

                    # Capture all <img> elements
                    img_elements = driver.find_elements(By.TAG_NAME, "img")

                    if not img_elements:
                        print("No image elements found on the page.")
                    else:
                        print(f"Found {len(img_elements)} image elements. Extracting images...")

                        for index, img_element in enumerate(img_elements, start=1):
                            img_src = img_element.get_attribute("src")
                            try:
                                if img_src.startswith("data:image"):
                                    img_data = base64.b64decode(img_src.split(",")[1])
                                else:
                                    img_data = requests.get(img_src, timeout=5).content

                                img = Image.open(BytesIO(img_data))
                                img = img.convert("RGB")
                                img.save(os.path.join(self.canvas_output_dir, f"image_{index}.jpg"), "JPEG")
                                print(f"Saved: image_{index}.jpg")
                            except Exception as e:
                                print(f"Failed to process image {index}: {e}")

                except Exception as e:
                    print(f"Error capturing canvas and images: {e}")
                    traceback.print_exc()

                finally:
                    driver.quit()

                    def capture_image_elements(self, driver):
                        """Captures all <img> elements and saves them as JPEGs."""
                        print("Searching for image elements...")
                        img_elements = driver.find_elements(By.TAG_NAME, "img")

                        if not img_elements:
                            print("No image elements found on the page.")
                            return

                        print(f"Found {len(img_elements)} image elements. Extracting images...")
                        for index, img_element in enumerate(img_elements, start=1):
                            try:
                                img_src = img_element.get_attribute("src")
                                if not img_src:
                                    continue
                                if img_src.startswith("data:image"):
                                    img_data = base64.b64decode(img_src.split(",")[1])
                                else:
                                    img_data = requests.get(img_src).content

                                img = Image.open(BytesIO(img_data)).convert("RGB")
                                img.save(os.path.join(self.canvas_output_dir, f"image_{index}.jpg"), "JPEG")
                                print(f"Saved image: image_{index}.jpg")
                            except Exception as e:
                                print(f"Failed to save image #{index}: {e}")


    def check_clickable_elements(self, url, page):
        """Checks all clickable elements (buttons, links) and logs results to a Word document."""

        doc = Document()
        doc.add_heading(f"Clickable Element Report for {url}", level=1)

        # Collect clickable elements
        clickable_elements = page.query_selector_all(
            'a, button, [role="button"], [onclick], [type="button"], [type="submit"]')

        for i, element in enumerate(clickable_elements):
            desc = "Unnamed element"
            try:
                # Try to get some description for logging
                text = element.inner_text(timeout=1000)
                if text and text.strip():
                    desc = text.strip()
                else:
                    outer_html = element.get_attribute('outerHTML')
                    desc = outer_html[:100] if outer_html else "Unnamed element"

                # Try to bring the element into view
                element.scroll_into_view_if_needed(timeout=3000)

                # Attempt to click
                element.click(timeout=3000)

                result = f"[{i + 1}] ✅ Click success: {desc}"
            except PlaywrightTimeoutError as te:
                result = f"[{i + 1}] ❌ Click failed: {desc} | Timeout Error: {str(te).splitlines()[0]}"
            except Exception as e:
                result = f"[{i + 1}] ❌ Click failed: {desc} | Reason: {str(e).splitlines()[0]}"

            # Add to Word document
            doc.add_paragraph(result)
