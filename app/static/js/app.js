document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("upload-form");
    const uploadResponse = document.getElementById("upload-response");
    const datasetList = document.getElementById("dataset-list");
    const refreshDatasetsBtn = document.getElementById("refresh-datasets");

    const trainForm = document.getElementById("train-form");
    const trainResponse = document.getElementById("train-response");

    const modelsList = document.getElementById("model-list");
    const refreshModelsBtn = document.getElementById("refresh-models");

    const predictForm = document.getElementById("predict-form");
    const predictResponse = document.getElementById("predict-response");

    const shapForm = document.getElementById("shap-form");
    const shapResponse = document.getElementById("shap-response");
    const shapImage = document.getElementById("shap-image");

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const fileInput = document.getElementById("dataset-file");
        if (!fileInput.files.length) {
            uploadResponse.textContent = "请先选择一个 CSV 文件。";
            return;
        }
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        uploadResponse.textContent = "上传中...";
        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            if (response.ok) {
                uploadResponse.textContent = `上传成功：${data.dataset_name}`;
                fileInput.value = "";
                loadDatasets();
            } else {
                uploadResponse.textContent = `上传失败：${data.detail || "未知错误"}`;
            }
        } catch (error) {
            uploadResponse.textContent = `上传失败：${error}`;
        }
    });

    async function loadDatasets() {
        datasetList.innerHTML = "加载中...";
        const response = await fetch("/api/datasets");
        const data = await response.json();
        datasetList.innerHTML = "";
        Object.entries(data.datasets || {}).forEach(([name, rows]) => {
            const li = document.createElement("li");
            li.textContent = `${name} - ${rows} 行`;
            datasetList.appendChild(li);
        });
    }

    refreshDatasetsBtn.addEventListener("click", loadDatasets);
    loadDatasets();

    trainForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        trainResponse.textContent = "训练中...";
        const payload = {
            dataset_name: document.getElementById("train-dataset").value,
            target_column: document.getElementById("train-target").value,
            model_name: document.getElementById("train-model-name").value,
            model_type: document.getElementById("train-model-type").value,
            test_size: parseFloat(document.getElementById("train-test-size").value),
            random_state: parseInt(document.getElementById("train-random-state").value, 10),
            model_params: {},
        };
        try {
            const response = await fetch("/api/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (response.ok) {
                trainResponse.textContent = `训练完成。准确率：${(data.metrics.accuracy * 100).toFixed(2)}%`;
                loadModels();
            } else {
                trainResponse.textContent = `训练失败：${data.detail || "未知错误"}`;
            }
        } catch (error) {
            trainResponse.textContent = `训练失败：${error}`;
        }
    });

    async function loadModels() {
        modelsList.innerHTML = "加载中...";
        const response = await fetch("/api/models");
        const data = await response.json();
        modelsList.innerHTML = "";
        Object.entries(data.models || {}).forEach(([name, info]) => {
            const li = document.createElement("li");
            li.textContent = `${name} - ${info.model_type} - Accuracy: ${(info.metrics.accuracy * 100).toFixed(2)}%`;
            modelsList.appendChild(li);
        });
    }

    refreshModelsBtn.addEventListener("click", loadModels);
    loadModels();

    predictForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        predictResponse.textContent = "预测中...";
        let features = {};
        try {
            features = JSON.parse(document.getElementById("predict-features").value || "{}");
        } catch (error) {
            predictResponse.textContent = "特征 JSON 格式不正确。";
            return;
        }
        const payload = {
            model_name: document.getElementById("predict-model-name").value,
            features,
        };
        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (response.ok) {
                predictResponse.textContent = `预测结果：${data.predictions.join(", ")}`;
            } else {
                predictResponse.textContent = `预测失败：${data.detail || "未知错误"}`;
            }
        } catch (error) {
            predictResponse.textContent = `预测失败：${error}`;
        }
    });

    shapForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        shapResponse.textContent = "生成中...";
        shapImage.style.display = "none";
        const payload = {
            model_name: document.getElementById("shap-model-name").value,
            sample_size: parseInt(document.getElementById("shap-sample-size").value, 10),
        };
        try {
            const response = await fetch("/api/shap", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (response.ok) {
                shapImage.src = `data:image/png;base64,${data.image_base64}`;
                shapImage.style.display = "block";
                shapResponse.textContent = "SHAP 图已生成。";
            } else {
                shapResponse.textContent = `生成失败：${data.detail || "未知错误"}`;
            }
        } catch (error) {
            shapResponse.textContent = `生成失败：${error}`;
        }
    });
});
