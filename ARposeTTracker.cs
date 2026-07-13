using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;
using Unity.Collections;
using UnityEngine.EventSystems;
using System.Globalization;
using UnityEngine.XR.ARSubsystems;

[System.Serializable]
public class SelectionData
{
    public List<int> selected; 
}

[System.Serializable]
public class PreprocessResponse
{
    public string status;
    public int total_images;
    public string session_id;
}

[System.Serializable]
public class ServerStatusResponse
{
    public string status;
    public string message;
}

[System.Serializable]
public class SlamPointData
{
    public float x;
    public float y;
    public float z;
    public float confidence;
}

[System.Serializable]
public class SlamPointPayload
{
    public string schema = "arpose_tracker_frame_points_v1";
    public int point_count;
    public List<SlamPointData> points = new List<SlamPointData>();
}

public class SegClick
{
    public float x;
    public float y;
    public int label;

    public SegClick(float xNorm, float yNorm, int pointLabel)
    {
        x = xNorm;
        y = yNorm;
        label = pointLabel;
    }
}

public class ARPoseTracker : MonoBehaviour
{
    [Header("AR 核心组件")]
    public Transform arCamera;
    public ARSession arSession;
    public ARCameraManager cameraManager;
    public ARPointCloudManager pointCloudManager;

    [Header("AR/SLAM 点云上传")]
    public bool uploadSlamPoints = true;
    public int maxSlamPointsPerFrame = 800;
    public int slamPointStride = 1;

    [Header("主 UI")]
    public Text debugText;
    public Button recordButton;
    public Text buttonText;

    [Header("人工审核 UI")]
    public GameObject reviewPanel;       
    public RawImage previewImage;        
    public Text imageIndexText;          
    public Text keepStatusText;          
    public Button btnPrev;               
    public Button btnNext;               
    public Button btnToggleKeep;         
    public Button btnSubmit;
    public Button btnCancel;             // 新增：退出/重拍按钮

    [Header("网络设置")]
    public string serverURL = "http://10.102.33.100:5000";
    public float sendInterval = 0.5f;

    private float timer = 0f;
    private bool isSending = false;
    private bool isRecording = false;
    private Texture2D cameraTexture;

    private List<Texture2D> downloadedPreviews = new List<Texture2D>();
    private List<List<SegClick>> frameClicks = new List<List<SegClick>>();
    private HashSet<int> approvedSeedFrames = new HashSet<int>();
    private int currentPreviewIndex = 0;
    private int capturedFrameCount = 0;
    private bool addForegroundPoint = true;
    private bool segmentationReady = false;
    private Texture2D currentDisplayTexture;
    private AspectRatioFitter previewAspectFitter;
    private const string CpuImageTransformName = "None";

    void Start()
    {
        if (reviewPanel != null) reviewPanel.SetActive(false);
        recordButton.onClick.AddListener(ToggleRecording);
        buttonText.text = "开始录制";

        btnPrev.onClick.AddListener(() => ChangePreview(-1));
        btnNext.onClick.AddListener(() => ChangePreview(1));
        btnToggleKeep.onClick.AddListener(TogglePointMode);
        btnSubmit.onClick.AddListener(SubmitSelection);
        
        // 绑定退出按钮
        if (btnCancel != null) btnCancel.onClick.AddListener(CancelReview);

        if (previewImage != null)
        {
            previewAspectFitter = previewImage.GetComponent<AspectRatioFitter>();
            if (previewAspectFitter == null) previewAspectFitter = previewImage.gameObject.AddComponent<AspectRatioFitter>();
            previewAspectFitter.aspectMode = AspectRatioFitter.AspectMode.FitInParent;
            previewImage.raycastTarget = true;

            EventTrigger trigger = previewImage.gameObject.GetComponent<EventTrigger>();
            if (trigger == null) trigger = previewImage.gameObject.AddComponent<EventTrigger>();
            EventTrigger.Entry entry = new EventTrigger.Entry();
            entry.eventID = EventTriggerType.PointerClick;
            entry.callback.AddListener((eventData) => OnPreviewClicked((PointerEventData)eventData));
            trigger.triggers.Add(entry);
        }
    }

    void ToggleRecording()
    {
        isRecording = !isRecording;

        if (isRecording)
        {
            segmentationReady = false;
            addForegroundPoint = true;
            approvedSeedFrames.Clear();
            buttonText.text = "结束录制并分割";
            buttonText.color = Color.red;
            StartCoroutine(SendCommand("/start_record"));
        }
        else
        {
            recordButton.gameObject.SetActive(false); 
            UpdateUI("采集结束，正在准备原图点选...", Color.yellow);
            StartCoroutine(RequestPreprocess());
        }
    }

    // ========== 更新：全局取消与重置 ==========
    void CancelReview()
    {
        // 1. 强行打断所有的前端状态
        isRecording = false;
        isSending = false;
        segmentationReady = false;
        addForegroundPoint = true;
        frameClicks.Clear();
        approvedSeedFrames.Clear();
        
        // 2. 隐藏审核面板（如果它正开着的话）
        if (reviewPanel != null) reviewPanel.SetActive(false);
        
        // 3. 恢复主界面的录制按钮
        recordButton.gameObject.SetActive(true);
        buttonText.text = "开始录制";
        buttonText.color = Color.white;
        
        UpdateUI("已强制重置，可随时开始新的录制", Color.white);

        // 4. 通知服务器中断当前任务并清空缓存
        StartCoroutine(SendCommand("/cancel_review"));
    }
    // ==========================================

    void Update()
    {
        if (ARSession.state != ARSessionState.SessionTracking) return;

        if (arCamera != null && isRecording)
        {
            Vector3 pos = arCamera.position;
            Quaternion quat = arCamera.rotation;
            UpdateUI($"[录制中] \nPos: {pos.x:F2}, {pos.y:F2}, {pos.z:F2}", Color.green);

            timer += Time.deltaTime;
            if (timer >= sendInterval && !isSending)
            {
                timer = 0f;
                XRCameraIntrinsics intrinsics = default(XRCameraIntrinsics);
                bool hasIntrinsics = cameraManager != null && cameraManager.TryGetIntrinsics(out intrinsics);
                if (cameraManager != null && cameraManager.TryAcquireLatestCpuImage(out XRCpuImage image))
                {
                    StartCoroutine(ProcessAndSendData(image, pos, quat, hasIntrinsics, intrinsics));
                }
            }
        }
    }

    IEnumerator RequestPreprocess()
    {
        using (UnityWebRequest www = UnityWebRequest.PostWwwForm(serverURL + "/preprocess", ""))
        {
            www.timeout = 120; 
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                PreprocessResponse response = JsonUtility.FromJson<PreprocessResponse>(www.downloadHandler.text);
                int totalImages = response.total_images;
                capturedFrameCount = totalImages;
                
                UpdateUI($"采集完成，共 {totalImages} 张。请在原图上点击物体前景/背景点进行监督分割。", Color.green);
                StartCoroutine(DownloadAllFrames(totalImages));
            }
            else
            {
                UpdateUI("准备分割失败: " + www.error, Color.red);
                // 失败时也把重拍按钮亮出来
                recordButton.gameObject.SetActive(true);
                buttonText.text = "开始录制";
                buttonText.color = Color.white;
            }
        }
    }

    IEnumerator DownloadAllFrames(int count)
    {
        downloadedPreviews.Clear();
        frameClicks.Clear();
        approvedSeedFrames.Clear();
        for (int i = 0; i < count; i++)
        {
            UpdateUI($"正在下载原图 {i + 1}/{count}...", Color.white);
            using (UnityWebRequest www = UnityWebRequestTexture.GetTexture(serverURL + $"/get_frame/{i}"))
            {
                yield return www.SendWebRequest();
                if (www.result == UnityWebRequest.Result.Success)
                {
                    downloadedPreviews.Add(DownloadHandlerTexture.GetContent(www));
                    frameClicks.Add(new List<SegClick>());
                }
            }
        }

        reviewPanel.SetActive(true);
        currentPreviewIndex = 0;
        RefreshReviewUI();
    }

    void ChangePreview(int step)
    {
        currentPreviewIndex += step;
        if (currentPreviewIndex < 0) currentPreviewIndex = downloadedPreviews.Count - 1;
        if (currentPreviewIndex >= downloadedPreviews.Count) currentPreviewIndex = 0;
        RefreshReviewUI();
    }

    void TogglePointMode()
    {
        addForegroundPoint = !addForegroundPoint;
        RefreshReviewUI();
    }

    void SetButtonLabel(Button button, string label)
    {
        if (button == null) return;
        Text text = button.GetComponentInChildren<Text>();
        if (text != null) text.text = label;
    }

    void RefreshReviewUI()
    {
        if (downloadedPreviews.Count == 0) return;
        Texture2D baseTexture = downloadedPreviews[currentPreviewIndex];
        if (previewAspectFitter != null && baseTexture.height > 0)
            previewAspectFitter.aspectRatio = (float)baseTexture.width / baseTexture.height;

        if (currentDisplayTexture != null && !downloadedPreviews.Contains(currentDisplayTexture))
            Destroy(currentDisplayTexture);
        currentDisplayTexture = segmentationReady ? baseTexture : BuildTextureWithClicks(baseTexture, currentPreviewIndex);
        previewImage.texture = currentDisplayTexture;
        imageIndexText.text = $"当前: {currentPreviewIndex + 1} / {downloadedPreviews.Count}";

        bool frameApproved = approvedSeedFrames.Contains(currentPreviewIndex);
        bool frameHasClicks = FrameHasLocalClicks(currentPreviewIndex);

        if (segmentationReady)
        {
            keepStatusText.text = "分割预览: 满意请提交；不满意继续点前景/背景";
            keepStatusText.color = Color.green;
            SetButtonLabel(btnSubmit, "确认生成");
        }
        else if (frameApproved)
        {
            keepStatusText.text = $"当前帧已作为种子；种子帧 {approvedSeedFrames.Count} 张";
            keepStatusText.color = Color.green;
            SetButtonLabel(btnSubmit, "运行分割");
        }
        else if (frameHasClicks)
        {
            keepStatusText.text = addForegroundPoint ? "检查当前预览，满意请确认本帧为种子" : "检查当前预览，满意请确认本帧为种子";
            keepStatusText.color = Color.yellow;
            SetButtonLabel(btnSubmit, "确认本帧");
        }
        else
        {
            keepStatusText.text = approvedSeedFrames.Count > 0
                ? $"可运行分割；也可继续点选更多种子帧，当前种子 {approvedSeedFrames.Count} 张"
                : (addForegroundPoint ? "点选模式: 前景点，先确认至少一帧种子" : "点选模式: 背景点，先确认至少一帧种子");
            keepStatusText.color = addForegroundPoint ? Color.green : Color.red;
            SetButtonLabel(btnSubmit, approvedSeedFrames.Count > 0 ? "运行分割" : "确认本帧");
        }
        SetButtonLabel(btnToggleKeep, addForegroundPoint ? "前景点" : "背景点");
    }

    bool FrameHasLocalClicks(int frameIndex)
    {
        return frameIndex >= 0 && frameIndex < frameClicks.Count && frameClicks[frameIndex].Count > 0;
    }

    void OnPreviewClicked(PointerEventData eventData)
    {
        if (reviewPanel == null || !reviewPanel.activeSelf || previewImage == null) return;

        RectTransform rect = previewImage.rectTransform;
        Vector2 localPoint;
        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(rect, eventData.position, eventData.pressEventCamera, out localPoint))
            return;

        Rect imageRect = GetDisplayedImageRect();
        if (!imageRect.Contains(localPoint)) return;

        float xNorm = Mathf.Clamp01((localPoint.x - imageRect.xMin) / imageRect.width);
        float yNorm = Mathf.Clamp01(1.0f - ((localPoint.y - imageRect.yMin) / imageRect.height));
        int label = addForegroundPoint ? 1 : 0;
        if (segmentationReady)
            segmentationReady = false;
        approvedSeedFrames.Remove(currentPreviewIndex);
        frameClicks[currentPreviewIndex].Add(new SegClick(xNorm, yNorm, label));
        RefreshReviewUI();
        StartCoroutine(SendSegPoint(currentPreviewIndex, xNorm, yNorm, label));
    }

    Rect GetDisplayedImageRect()
    {
        Rect r = previewImage.rectTransform.rect;
        Texture tex = previewImage.texture;
        if (tex == null || tex.height <= 0) return r;

        float imageAspect = (float)tex.width / tex.height;
        float rectAspect = r.width / r.height;
        if (rectAspect > imageAspect)
        {
            float width = r.height * imageAspect;
            float x = r.xMin + (r.width - width) * 0.5f;
            return new Rect(x, r.yMin, width, r.height);
        }
        else
        {
            float height = r.width / imageAspect;
            float y = r.yMin + (r.height - height) * 0.5f;
            return new Rect(r.xMin, y, r.width, height);
        }
    }

    Texture2D BuildTextureWithClicks(Texture2D source, int frameIndex)
    {
        Texture2D tex = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false);
        tex.SetPixels32(source.GetPixels32());

        if (frameIndex >= 0 && frameIndex < frameClicks.Count)
        {
            foreach (SegClick click in frameClicks[frameIndex])
            {
                int x = Mathf.RoundToInt(click.x * (source.width - 1));
                int y = Mathf.RoundToInt((1.0f - click.y) * (source.height - 1));
                DrawClickMarker(tex, x, y, click.label == 1 ? Color.green : Color.red);
            }
        }

        tex.Apply();
        return tex;
    }

    void DrawClickMarker(Texture2D tex, int cx, int cy, Color color)
    {
        int radius = Mathf.Max(6, Mathf.RoundToInt(Mathf.Min(tex.width, tex.height) * 0.012f));
        Color outline = Color.white;
        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                int x = cx + dx;
                int y = cy + dy;
                if (x < 0 || y < 0 || x >= tex.width || y >= tex.height) continue;
                float d = Mathf.Sqrt(dx * dx + dy * dy);
                if (d <= radius)
                    tex.SetPixel(x, y, color);
                if (Mathf.Abs(d - radius) < 1.8f)
                    tex.SetPixel(x, y, outline);
            }
        }
    }

    IEnumerator SendSegPoint(int frameIndex, float xNorm, float yNorm, int label)
    {
        string json = string.Format(
            CultureInfo.InvariantCulture,
            "{{\"frame_index\":{0},\"x\":{1:F6},\"y\":{2:F6},\"label\":{3},\"normalized\":true}}",
            frameIndex,
            xNorm,
            yNorm,
            label
        );
        UnityWebRequest www = new UnityWebRequest(serverURL + "/add_seg_point", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
        www.uploadHandler = new UploadHandlerRaw(bodyRaw);
        www.downloadHandler = new DownloadHandlerBuffer();
        www.SetRequestHeader("Content-Type", "application/json");
        yield return www.SendWebRequest();

        if (www.result == UnityWebRequest.Result.Success)
        {
            UpdateUI($"已添加{(label == 1 ? "前景" : "背景")}点: frame {frameIndex}, ({xNorm:F2},{yNorm:F2})", Color.green);
            StartCoroutine(UpdateCurrentFrameSegmentationPreview(frameIndex));
        }
        else
            UpdateUI("添加分割点失败: " + www.error, Color.red);
    }

    IEnumerator UpdateCurrentFrameSegmentationPreview(int frameIndex)
    {
        using (UnityWebRequest seg = UnityWebRequest.PostWwwForm(serverURL + $"/segment_frame/{frameIndex}", ""))
        {
            seg.timeout = 120;
            yield return seg.SendWebRequest();
            if (seg.result != UnityWebRequest.Result.Success)
            {
                UpdateUI("当前帧分割失败: " + seg.error, Color.red);
                yield break;
            }
        }

        using (UnityWebRequest www = UnityWebRequestTexture.GetTexture(serverURL + $"/get_prompt_preview/{frameIndex}?t={Time.realtimeSinceStartup}"))
        {
            yield return www.SendWebRequest();
            if (www.result == UnityWebRequest.Result.Success && frameIndex >= 0 && frameIndex < downloadedPreviews.Count)
            {
                downloadedPreviews[frameIndex] = DownloadHandlerTexture.GetContent(www);
                if (frameIndex == currentPreviewIndex)
                    RefreshReviewUI();
            }
        }
    }

    void SubmitSelection()
    {
        if (!segmentationReady)
        {
            if (!approvedSeedFrames.Contains(currentPreviewIndex) && FrameHasLocalClicks(currentPreviewIndex))
            {
                UpdateUI("正在确认当前帧为视频传播种子...", Color.yellow);
                StartCoroutine(ApproveCurrentSeedFrame(currentPreviewIndex));
                return;
            }

            if (approvedSeedFrames.Count == 0)
            {
                UpdateUI("请先在至少一帧上点选，并确认该帧为种子。", Color.red);
                return;
            }

            UpdateUI($"正在使用 {approvedSeedFrames.Count} 张种子帧进行整段视频分割...", Color.yellow);
            StartCoroutine(RunSegmentationAndPreview());
            return;
        }

        reviewPanel.SetActive(false);
        UpdateUI("分割已确认，正在检查采集视角覆盖...", Color.yellow);
        StartCoroutine(CheckInputQcThenGenerate());
    }

    IEnumerator ApproveCurrentSeedFrame(int frameIndex)
    {
        using (UnityWebRequest www = UnityWebRequest.PostWwwForm(serverURL + $"/approve_seed_frame/{frameIndex}", ""))
        {
            www.timeout = 120;
            yield return www.SendWebRequest();
            if (www.result == UnityWebRequest.Result.Success)
            {
                approvedSeedFrames.Add(frameIndex);
                UpdateUI($"已确认第 {frameIndex + 1} 帧为种子。可继续确认其它帧，或再次提交运行分割。", Color.green);
                RefreshReviewUI();
            }
            else
            {
                UpdateUI("确认种子帧失败: " + www.error, Color.red);
            }
        }
    }

    IEnumerator RunSegmentationAndPreview()
    {
        using (UnityWebRequest seg = UnityWebRequest.PostWwwForm(serverURL + "/run_segmentation", ""))
        {
            seg.timeout = 600;
            yield return seg.SendWebRequest();
            if (seg.result != UnityWebRequest.Result.Success)
            {
                UpdateUI("视频分割失败: " + seg.error, Color.red);
                recordButton.gameObject.SetActive(true);
                buttonText.text = "开始录制";
                buttonText.color = Color.white;
                yield break;
            }
        }

        yield return StartCoroutine(DownloadSegmentedPreviews(downloadedPreviews.Count));
        segmentationReady = true;
        currentPreviewIndex = 0;
        reviewPanel.SetActive(true);
        RefreshReviewUI();
        UpdateUI("分割完成，请检查 preview，确认后再次提交生成。", Color.green);
    }

    IEnumerator DownloadSegmentedPreviews(int count)
    {
        downloadedPreviews.Clear();
        frameClicks.Clear();
        for (int i = 0; i < count; i++)
        {
            UpdateUI($"正在下载分割 preview {i + 1}/{count}...", Color.white);
            using (UnityWebRequest www = UnityWebRequestTexture.GetTexture(serverURL + $"/get_preview/{i}"))
            {
                yield return www.SendWebRequest();
                if (www.result == UnityWebRequest.Result.Success)
                {
                    downloadedPreviews.Add(DownloadHandlerTexture.GetContent(www));
                    frameClicks.Add(new List<SegClick>());
                }
            }
        }
    }

    IEnumerator SendGenerateFromAllFrames()
    {
        SelectionData data = new SelectionData();
        data.selected = new List<int>();
        for (int i = 0; i < capturedFrameCount; i++)
        {
            data.selected.Add(i);
        }

        string jsonPayload = JsonUtility.ToJson(data);
        yield return StartCoroutine(SendGenerateCommand(jsonPayload));
    }

    IEnumerator CheckInputQcThenGenerate()
    {
        SelectionData data = new SelectionData();
        data.selected = new List<int>();
        for (int i = 0; i < capturedFrameCount; i++)
        {
            data.selected.Add(i);
        }

        string jsonPayload = JsonUtility.ToJson(data);
        UnityWebRequest qc = new UnityWebRequest(serverURL + "/input_qc", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
        qc.uploadHandler = new UploadHandlerRaw(bodyRaw);
        qc.downloadHandler = new DownloadHandlerBuffer();
        qc.SetRequestHeader("Content-Type", "application/json");
        qc.timeout = 120;
        yield return qc.SendWebRequest();

        if (qc.result != UnityWebRequest.Result.Success)
        {
            UpdateUI("输入检查失败: " + ExtractServerMessage(qc), Color.red);
            recordButton.gameObject.SetActive(true);
            buttonText.text = "开始新录制";
            buttonText.color = Color.white;
            yield break;
        }

        string responseText = qc.downloadHandler.text;
        ServerStatusResponse response = JsonUtility.FromJson<ServerStatusResponse>(responseText);
        if (response != null && response.status == "warning")
        {
            string msg = string.IsNullOrEmpty(response.message) ? "输入视角覆盖不足，请补采或重拍。" : response.message;
            UpdateUI(msg, Color.red);
            recordButton.gameObject.SetActive(true);
            buttonText.text = "重新录制";
            buttonText.color = Color.white;
            yield break;
        }

        UpdateUI("输入检查通过，服务器正在重建 Mesh (需耐心等待)！", Color.yellow);
        yield return StartCoroutine(SendGenerateCommand(jsonPayload));
    }

    IEnumerator SendGenerateCommand(string json)
    {
        UnityWebRequest www = new UnityWebRequest(serverURL + "/generate", "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
        www.uploadHandler = new UploadHandlerRaw(bodyRaw);
        www.downloadHandler = new DownloadHandlerBuffer();
        www.SetRequestHeader("Content-Type", "application/json");
        
        www.timeout = 600; 

        yield return www.SendWebRequest();

        if (www.result == UnityWebRequest.Result.Success)
        {
            string msg = ExtractServerMessage(www);
            UpdateUI(string.IsNullOrEmpty(msg) ? "重建完成！请在服务器 output 文件夹查看 Mesh" : msg, Color.green);
            recordButton.gameObject.SetActive(true);
            buttonText.text = "开始新录制";
            buttonText.color = Color.white;
        }
        else
        {
            UpdateUI("生成失败: " + ExtractServerMessage(www), Color.red);
            recordButton.gameObject.SetActive(true);
            buttonText.text = "开始新录制";
            buttonText.color = Color.white;
        }
    }

    IEnumerator ProcessAndSendData(XRCpuImage image, Vector3 pos, Quaternion quat, bool hasIntrinsics, XRCameraIntrinsics intrinsics)
    {
        isSending = true;
        var conversionParams = new UnityEngine.XR.ARSubsystems.XRCpuImage.ConversionParams
        {
            inputRect = new RectInt(0, 0, image.width, image.height),
            outputDimensions = new Vector2Int(image.width, image.height), // 已恢复全分辨率
            outputFormat = TextureFormat.RGBA32,
            transformation = UnityEngine.XR.ARSubsystems.XRCpuImage.Transformation.None
        };
        int cpuImageWidth = image.width;
        int cpuImageHeight = image.height;

        int size = image.GetConvertedDataSize(conversionParams);
        var buffer = new NativeArray<byte>(size, Allocator.Temp);
        image.Convert(conversionParams, buffer);
        image.Dispose();

        if (cameraTexture == null || cameraTexture.width != conversionParams.outputDimensions.x || cameraTexture.height != conversionParams.outputDimensions.y)
            cameraTexture = new Texture2D(conversionParams.outputDimensions.x, conversionParams.outputDimensions.y, TextureFormat.RGBA32, false);

        cameraTexture.LoadRawTextureData(buffer);
        cameraTexture.Apply();
        buffer.Dispose();

        byte[] imageBytes = cameraTexture.EncodeToJPG(75);
        WWWForm form = new WWWForm();
        Vector3 rot = quat.eulerAngles;
        form.AddField("pos_x", FloatString(pos.x));
        form.AddField("pos_y", FloatString(pos.y));
        form.AddField("pos_z", FloatString(pos.z));
        form.AddField("rot_x", FloatString(rot.x));
        form.AddField("rot_y", FloatString(rot.y));
        form.AddField("rot_z", FloatString(rot.z));
        form.AddField("quat_x", FloatString(quat.x));
        form.AddField("quat_y", FloatString(quat.y));
        form.AddField("quat_z", FloatString(quat.z));
        form.AddField("quat_w", FloatString(quat.w));
        form.AddField("image_width", IntString(cameraTexture.width));
        form.AddField("image_height", IntString(cameraTexture.height));
        form.AddField("cpu_image_width", IntString(cpuImageWidth));
        form.AddField("cpu_image_height", IntString(cpuImageHeight));
        form.AddField("image_transform", CpuImageTransformName);
        if (hasIntrinsics)
        {
            form.AddField("fx", FloatString(intrinsics.focalLength.x));
            form.AddField("fy", FloatString(intrinsics.focalLength.y));
            form.AddField("cx", FloatString(intrinsics.principalPoint.x));
            form.AddField("cy", FloatString(intrinsics.principalPoint.y));
            form.AddField("intrinsic_width", IntString(intrinsics.resolution.x));
            form.AddField("intrinsic_height", IntString(intrinsics.resolution.y));
        }
        string slamPointsJson = BuildSlamPointsJson();
        if (!string.IsNullOrEmpty(slamPointsJson))
        {
            form.AddField("slam_points_schema", "arpose_tracker_frame_points_v1");
            form.AddField("slam_points_json", slamPointsJson);
        }
        form.AddBinaryData("image", imageBytes, "frame.jpg", "image/jpeg");

        using (UnityWebRequest www = UnityWebRequest.Post(serverURL + "/upload", form))
        {
            yield return www.SendWebRequest();
        }
        isSending = false;
    }

    string BuildSlamPointsJson()
    {
        if (!uploadSlamPoints || pointCloudManager == null || maxSlamPointsPerFrame <= 0)
            return "";

        SlamPointPayload payload = new SlamPointPayload();
        int stride = Mathf.Max(1, slamPointStride);
        int seen = 0;

        foreach (ARPointCloud cloud in pointCloudManager.trackables)
        {
            if (!cloud.positions.HasValue)
                continue;

            NativeArray<Vector3> positions = cloud.positions.Value;
            for (int i = 0; i < positions.Length; i++)
            {
                if ((seen % stride) == 0)
                {
                    Vector3 p = positions[i];
                    payload.points.Add(new SlamPointData
                    {
                        x = p.x,
                        y = p.y,
                        z = p.z,
                        confidence = 1.0f
                    });
                    if (payload.points.Count >= maxSlamPointsPerFrame)
                        break;
                }
                seen++;
            }

            if (payload.points.Count >= maxSlamPointsPerFrame)
                break;
        }

        payload.point_count = payload.points.Count;
        return payload.point_count > 0 ? JsonUtility.ToJson(payload) : "";
    }

    IEnumerator SendCommand(string endpoint)
    {
        using (UnityWebRequest www = UnityWebRequest.PostWwwForm(serverURL + endpoint, ""))
            yield return www.SendWebRequest();
    }
    
    void UpdateUI(string text, Color color)
    {
        if (debugText != null) { debugText.text = text; debugText.color = color; }
    }

    static string FloatString(float value)
    {
        return value.ToString("R", CultureInfo.InvariantCulture);
    }

    static string IntString(int value)
    {
        return value.ToString(CultureInfo.InvariantCulture);
    }

    static string ExtractServerMessage(UnityWebRequest request)
    {
        if (request == null) return "";
        string body = request.downloadHandler != null ? request.downloadHandler.text : "";
        if (!string.IsNullOrEmpty(body))
        {
            try
            {
                ServerStatusResponse response = JsonUtility.FromJson<ServerStatusResponse>(body);
                if (response != null && !string.IsNullOrEmpty(response.message))
                    return response.message;
            }
            catch
            {
                // Fall back to Unity's transport error below.
            }
        }
        return string.IsNullOrEmpty(request.error) ? body : request.error;
    }
}
