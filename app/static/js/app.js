document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("upload-form");
    const uploadResponse = document.getElementById("upload-response");
    const refreshDatasetsBtn = document.getElementById("refresh-datasets");
    const datasetTableBody = document.getElementById("dataset-table-body");
    const datasetDetail = document.getElementById("dataset-detail");
    const datasetDetailContent = datasetDetail?.querySelector(".detail-content");

    const trainDatasetSelect = document.getElementById("train-dataset");
    const targetSuggestions = document.getElementById("target-suggestions");
    
    const trainForm = document.getElementById("train-form");
    const trainResponse = document.getElementById("train-response");
    const trainModelParams = document.getElementById("train-model-params");

    const refreshModelsBtn = document.getElementById("refresh-models");
    const modelTableBody = document.getElementById("model-table-body");
    const modelDetail = document.getElementById("model-detail");
    const modelDetailContent = modelDetail?.querySelector(".detail-content");

    const predictForm = document.getElementById("predict-form");
    const predictResponse = document.getElementById("predict-response");
    const predictModelSelect = document.getElementById("predict-model-name");

    const shapForm = document.getElementById("shap-form");
    const shapResponse = document.getElementById("shap-response");
    const shapImage = document.getElementById("shap-image");
    const shapModelSelect = document.getElementById("shap-model-name");

    const datasetCache = new Map();
    let lastSelectedDataset = "";
    let lastSelectedModel = "";

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
                lastSelectedDataset = data.dataset_name;
                loadDatasets();
            } else {
                uploadResponse.textContent = `上传失败：${data.detail || "未知错误"}`;
            }
        } catch (error) {
            uploadResponse.textContent = `上传失败：${error}`;
        }
    });

    async function loadDatasets() {
        datasetTableBody.innerHTML = "<tr><td colspan='5'>加载中...</td></tr>";
        datasetCache.clear();
        const response = await fetch("/api/datasets");
        const data = await response.json();
        const datasets = data.datasets || [];
        datasetTableBody.innerHTML = "";
        trainDatasetSelect.innerHTML = '<option value="" disabled selected>请选择数据集</option>';
        if (!datasets.length) {
            datasetTableBody.innerHTML = "<tr><td colspan='5'>暂未上传数据集</td></tr>";
            datasetDetail.hidden = true;
        }
        datasets.forEach((dataset) => {
            datasetCache.set(dataset.name, dataset);
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${dataset.name}</td>
                <td>${dataset.rows}</td>
                <td>${dataset.columns}</td>
                <td>${dataset.target_suggestions?.join(", ") || "-"}</td>
                <td class="table-actions"></td>
            `;

            const actionsCell = row.querySelector(".table-actions");
            const viewBtn = document.createElement("button");
            viewBtn.type = "button";
            viewBtn.textContent = "查看详情";
            viewBtn.classList.add("secondary");
            viewBtn.addEventListener("click", () => showDatasetDetail(dataset.name));

            const deleteBtn = document.createElement("button");
            deleteBtn.type = "button";
            deleteBtn.textContent = "删除";
            deleteBtn.classList.add("danger");
            deleteBtn.addEventListener("click", () => deleteDataset(dataset.name));

            actionsCell.appendChild(viewBtn);
            actionsCell.appendChild(deleteBtn);

            datasetTableBody.appendChild(row);

            const option = document.createElement("option");
            option.value = dataset.name;
            option.textContent = dataset.name;
            trainDatasetSelect.appendChild(option);
        });

        if (lastSelectedDataset && datasetCache.has(lastSelectedDataset)) {
            trainDatasetSelect.value = lastSelectedDataset;
            const summary = datasetCache.get(lastSelectedDataset);
            setTargetSuggestions(summary?.target_suggestions || []);
            showDatasetDetail(lastSelectedDataset);
        } else {
            trainDatasetSelect.value = "";
            setTargetSuggestions([]);
            if (datasetDetailContent) {
                datasetDetail.hidden = true;
                datasetDetailContent.innerHTML = "";
            }
        }
    }

    refreshDatasetsBtn.addEventListener("click", loadDatasets);
    loadDatasets();

    async function showDatasetDetail(name) {
        if (!datasetDetailContent) return;
        datasetDetail.hidden = false;
        datasetDetailContent.innerHTML = "加载中...";
        try {
            const response = await fetch(`/api/datasets/${encodeURIComponent(name)}`);
            const data = await response.json();
            if (!response.ok) {
                datasetDetailContent.textContent = data.detail || "加载失败";
                return;
            }

            datasetDetailContent.innerHTML = renderDatasetProfile(data);

            if (trainDatasetSelect.value === data.name) {
                setTargetSuggestions(data.column_details.map((col) => col.name));
            }
        } catch (error) {
            datasetDetailContent.textContent = `加载失败：${error}`;
        }
    }

    function renderDatasetProfile(profile) {
        const columnDetails = Array.isArray(profile.column_details) ? profile.column_details : [];
        const previewRows = Array.isArray(profile.preview) ? profile.preview : [];

        const columnsTable = columnDetails
            .map(
                (column) => `
                <tr>
                    <td>${column.name}</td>
                    <td>${column.dtype}</td>
                    <td>${column.missing}</td>
                    <td>${column.unique_values}</td>
                </tr>
            `
            )
            .join("");

        const previewTable = previewRows
            .map((row, index) => {
                const cells = Object.values(row)
                    .map((value) => `<td>${value ?? ""}</td>`)
                    .join("");
                return `<tr><td>${index + 1}</td>${cells}</tr>`;
            })
            .join("");

        const previewHeader = previewRows[0]
            ? Object.keys(previewRows[0])
                  .map((column) => `<th>${column}</th>`)
                  .join("")
            : "";

        return `
            <p><strong>${profile.name}</strong> — 共 ${profile.rows} 行 ${profile.columns} 列</p>
            <h4>字段概览</h4>
            <table>
                <thead>
                    <tr>
                        <th>字段</th>
                        <th>类型</th>
                        <th>缺失值</th>
                        <th>唯一值数量</th>
                    </tr>
                </thead>
                <tbody>${columnsTable}</tbody>
            </table>
            <h4>数据预览</h4>
            <table>
                <thead>
                    <tr><th>#</th>${previewHeader}</tr>
                </thead>
                <tbody>${previewTable || "<tr><td colspan='99'>无可用数据</td></tr>"}</tbody>
            </table>
        `;
    }

    async function deleteDataset(name) {
        if (!window.confirm(`确定要删除数据集 ${name} 吗？`)) {
            return;
        }
        try {
            const response = await fetch(`/api/datasets/${encodeURIComponent(name)}`, {
                method: "DELETE",
            });
            if (response.ok) {
                uploadResponse.textContent = `数据集 ${name} 已删除。`;
                datasetDetail.hidden = true;
                if (lastSelectedDataset === name) {
                    lastSelectedDataset = "";
                }
                loadDatasets();
            } else {
                const data = await response.json();
                uploadResponse.textContent = `删除失败：${data.detail || "未知错误"}`;
            }
        } catch (error) {
            uploadResponse.textContent = `删除失败：${error}`;
        }
    }

    function setTargetSuggestions(columns = []) {
        targetSuggestions.innerHTML = "";
        columns.forEach((column) => {
            const option = document.createElement("option");
            option.value = column;
            targetSuggestions.appendChild(option);
        });
    }

    trainDatasetSelect.addEventListener("change", (event) => {
        const datasetName = event.target.value;
        const summary = datasetCache.get(datasetName);
        if (summary) {
            setTargetSuggestions(summary.target_suggestions);
        }
        if (datasetName) {
            lastSelectedDataset = datasetName;
            showDatasetDetail(datasetName);
        }
    });

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
        if (trainModelParams.value.trim()) {
            try {
                payload.model_params = JSON.parse(trainModelParams.value);
            } catch (error) {
                trainResponse.textContent = "模型参数 JSON 格式不正确";
                return;
            }
        }
        lastSelectedModel = payload.model_name;
        try {
            const response = await fetch("/api/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (response.ok) {
                const metrics = data.metrics || {};
                trainResponse.innerHTML = renderMetricsSummary(metrics);
                loadModels();
            } else {
                trainResponse.textContent = `训练失败：${data.detail || "未知错误"}`;
            }
        } catch (error) {
            trainResponse.textContent = `训练失败：${error}`;
        }
    });

    async function loadModels() {
        modelTableBody.innerHTML = "<tr><td colspan='5'>加载中...</td></tr>";
        const response = await fetch("/api/models");
        const data = await response.json();
        const models = data.models || [];
        modelTableBody.innerHTML = "";
        predictModelSelect.innerHTML = '<option value="" disabled selected>请选择模型</option>';
        shapModelSelect.innerHTML = '<option value="" disabled selected>请选择模型</option>';
        if (!models.length) {
            modelTableBody.innerHTML = "<tr><td colspan='5'>暂未训练模型</td></tr>";
            modelDetail.hidden = true;
        }
        models.forEach((model) => {
            const accuracy = model.metrics?.accuracy ?? 0;
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${model.name}</td>
                <td>${model.model_type}</td>
                <td>${(accuracy * 100).toFixed(2)}%</td>
                <td>${(model.training_time_seconds ?? 0).toFixed(2)}</td>
                <td class="table-actions"></td>
            `;

            const actionsCell = row.querySelector(".table-actions");
            const viewBtn = document.createElement("button");
            viewBtn.type = "button";
            viewBtn.textContent = "查看详情";
            viewBtn.classList.add("secondary");
            viewBtn.addEventListener("click", () => showModelDetail(model.name));

            const deleteBtn = document.createElement("button");
            deleteBtn.type = "button";
            deleteBtn.textContent = "删除";
            deleteBtn.classList.add("danger");
            deleteBtn.addEventListener("click", () => deleteModel(model.name));

            actionsCell.appendChild(viewBtn);
            actionsCell.appendChild(deleteBtn);

            modelTableBody.appendChild(row);

            const option = document.createElement("option");
            option.value = model.name;
            option.textContent = model.name;
            predictModelSelect.appendChild(option);

            const shapOption = option.cloneNode(true);
            shapModelSelect.appendChild(shapOption);
        });

        if (lastSelectedModel) {
            const hasPredict = [...predictModelSelect.options].some(
                (option) => option.value === lastSelectedModel,
            );
            predictModelSelect.value = hasPredict ? lastSelectedModel : "";

            const hasShap = [...shapModelSelect.options].some(
                (option) => option.value === lastSelectedModel,
            );
            shapModelSelect.value = hasShap ? lastSelectedModel : "";
        } else {
            predictModelSelect.value = "";
            shapModelSelect.value = "";
        }
    }

    refreshModelsBtn.addEventListener("click", loadModels);
    loadModels();

    async function showModelDetail(name) {
        if (!modelDetailContent) return;
        modelDetail.hidden = false;
        modelDetailContent.innerHTML = "加载中...";
        try {
            const response = await fetch(`/api/models/${encodeURIComponent(name)}`);
            const data = await response.json();
            if (!response.ok) {
                modelDetailContent.textContent = data.detail || "加载失败";
                return;
            }
            modelDetailContent.innerHTML = renderModelDetail(data);
        } catch (error) {
            modelDetailContent.textContent = `加载失败：${error}`;
        }
    }

    function renderMetricsSummary(metrics) {
        const metricKeys = ["accuracy", "f1_weighted", "balanced_accuracy", "log_loss"];
        const items = metricKeys
            .filter((key) => key in metrics)
            .map((key) => {
                const value = metrics[key];
                const formatted = typeof value === "number" ? value.toFixed(4) : value;
                return `<li>${key}: ${formatted}</li>`;
            })
            .join("");
        return `<strong>训练完成：</strong><ul>${items || "<li>暂无指标</li>"}</ul>`;
    }

    function renderModelDetail(model) {
        const metrics = model.metrics || {};
        const metricRows = Object.entries(metrics)
            .map(
                ([key, value]) => `
                <tr>
                    <td>${key}</td>
                    <td>${typeof value === "number" ? value.toFixed(4) : value}</td>
                </tr>
            `
            )
            .join("");

        const labels = model.class_labels || model.report?.labels || [];
        const confusionMatrix = model.report?.confusion_matrix || [];
        const matrixRows = confusionMatrix
            .map((row, index) => {
                const cells = row.map((value) => `<td>${value}</td>`).join("");
                const label = labels[index] ?? index;
                return `<tr><th>${label}</th>${cells}</tr>`;
            })
            .join("");

        const matrixHeader = labels
            .map((label) => `<th>${label}</th>`)
            .join("");

        return `
            <p><strong>${model.name}</strong> · ${model.model_type}</p>
            <p>训练耗时：${(model.training_time_seconds ?? 0).toFixed(2)} 秒 ｜ 数据集分层：${model.stratified_split ? "是" : "否"}</p>
            <h4>核心指标</h4>
            <table>
                <tbody>${metricRows}</tbody>
            </table>
            <h4>混淆矩阵</h4>
            <table>
                <thead><tr><th></th>${matrixHeader}</tr></thead>
                <tbody>${matrixRows || "<tr><td colspan='99'>暂无数据</td></tr>"}</tbody>
            </table>
            <h4>分类报告</h4>
            <pre>${JSON.stringify(model.report?.classification_report || {}, null, 2)}</pre>
            <p>特征列：${(model.feature_names || []).join(", ")}</p>
        `;
    }

    async function deleteModel(name) {
        if (!window.confirm(`确定要删除模型 ${name} 吗？`)) {
            return;
        }
        try {
            const response = await fetch(`/api/models/${encodeURIComponent(name)}`, {
                method: "DELETE",
            });
            if (response.ok) {
                trainResponse.textContent = `模型 ${name} 已删除。`;
                modelDetail.hidden = true;
                if (lastSelectedModel === name) {
                    lastSelectedModel = "";
                }
                loadModels();
            } else {
                const data = await response.json();
                trainResponse.textContent = `删除失败：${data.detail || "未知错误"}`;
            }
        } catch (error) {
            trainResponse.textContent = `删除失败：${error}`;
        }
    }

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
            model_name: predictModelSelect.value,
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
                const probabilityText = data.probabilities
                    ? `\n概率：${JSON.stringify(data.probabilities)}`
                    : "";
                predictResponse.textContent = `预测结果：${data.predictions.join(", ")}${probabilityText}`;
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
            model_name: shapModelSelect.value,
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
