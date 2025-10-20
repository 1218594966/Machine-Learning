document.addEventListener("DOMContentLoaded", () => {
    const defaultOptions = {
        numeric_imputation: "most_frequent",
        categorical_imputation: "most_frequent",
        numeric_scaling: "none",
        categorical_encoding: "one_hot",
        constant_fill_value: "",
    };

    const state = {
        datasets: new Map(),
        datasetProfiles: new Map(),
        selections: new Map(),
        selectedDataset: "",
        lastModel: "",
    };

    // ------------------------------------------------------------------
    // DOM references
    // ------------------------------------------------------------------
    const uploadForm = document.getElementById("upload-form");
    const uploadResponse = document.getElementById("upload-response");
    const uploadPreview = document.getElementById("upload-preview");
    const uploadPreviewHead = document.getElementById("upload-preview-head");
    const uploadPreviewBody = document.getElementById("upload-preview-body");

    const refreshDatasetsBtn = document.getElementById("refresh-datasets");
    const datasetTableBody = document.getElementById("dataset-table-body");
    const datasetDetail = document.getElementById("dataset-detail");
    const datasetDetailContent = datasetDetail?.querySelector(".detail-content");

    const featureDatasetSelect = document.getElementById("feature-dataset");
    const targetColumnContainer = document.getElementById("target-column-container");
    const featureColumnContainer = document.getElementById("feature-column-container");
    const featureResponse = document.getElementById("feature-response");
    const previewButton = document.getElementById("preview-transformation");
    const previewTable = document.getElementById("preview-table");
    const previewTableHead = document.getElementById("preview-table-head");
    const previewTableBody = document.getElementById("preview-table-body");

    const numericImputation = document.getElementById("numeric-imputation");
    const categoricalImputation = document.getElementById("categorical-imputation");
    const numericScaling = document.getElementById("numeric-scaling");
    const categoricalEncoding = document.getElementById("categorical-encoding");
    const constantFillWrapper = document.getElementById("constant-fill-wrapper");
    const constantFillValue = document.getElementById("constant-fill-value");

    const trainDatasetLabel = document.getElementById("train-dataset-label");
    const trainForm = document.getElementById("train-form");
    const trainModeSelect = document.getElementById("train-mode");
    const trainAlgorithmSelect = document.getElementById("train-algorithm");
    const trainTestSizeInput = document.getElementById("train-test-size");
    const trainRandomStateInput = document.getElementById("train-random-state");
    const trainModelParams = document.getElementById("train-model-params");
    const trainResponse = document.getElementById("train-response");

    const refreshModelsBtn = document.getElementById("refresh-models");
    const modelTableBody = document.getElementById("model-table-body");
    const modelDetail = document.getElementById("model-detail");
    const modelDetailContent = modelDetail?.querySelector(".detail-content");

    const predictForm = document.getElementById("predict-form");
    const predictResponse = document.getElementById("predict-response");
    const predictModelSelect = document.getElementById("predict-model-name");
    const predictFeatures = document.getElementById("predict-features");

    const shapForm = document.getElementById("shap-form");
    const shapResponse = document.getElementById("shap-response");
    const shapImage = document.getElementById("shap-image");
    const shapModelSelect = document.getElementById("shap-model-name");

    // ------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------
    const safeJson = async (response) => {
        const text = await response.text();
        if (!text) {
            return {};
        }
        try {
            return JSON.parse(text);
        } catch (error) {
            return { detail: text || response.statusText };
        }
    };

    const getSelection = (datasetName) => {
        if (!state.selections.has(datasetName)) {
            state.selections.set(datasetName, {
                target: "",
                features: new Set(),
                options: { ...defaultOptions },
            });
        }
        return state.selections.get(datasetName);
    };

    const toggleConstantField = () => {
        const requiresConstant =
            numericImputation.value === "constant" ||
            categoricalImputation.value === "constant";
        constantFillWrapper.hidden = !requiresConstant;
        constantFillValue.required = requiresConstant;
        if (!requiresConstant) {
            constantFillValue.value = constantFillValue.value.trim();
        }
    };

    const resetPreview = () => {
        previewTableHead.innerHTML = "";
        previewTableBody.innerHTML = "";
        previewTable.hidden = true;
    };

    const renderPreviewTable = (headElement, bodyElement, columns, rows) => {
        headElement.innerHTML = columns.map((name) => `<th>${name}</th>`).join("");
        if (!rows.length) {
            bodyElement.innerHTML = "<tr><td colspan='99'>暂无数据</td></tr>";
            return;
        }
        const bodyHtml = rows
            .map((row) => {
                const cells = columns
                    .map((column) => `<td>${row[column] ?? ""}</td>`)
                    .join("");
                return `<tr>${cells}</tr>`;
            })
            .join("");
        bodyElement.innerHTML = bodyHtml;
    };

    const renderDatasetProfile = (profile) => {
        const columnRows = profile.column_details
            .map(
                (column) => `
                <tr>
                    <td>${column.name}</td>
                    <td>${column.dtype}</td>
                    <td>${column.missing}</td>
                    <td>${column.unique_values}</td>
                </tr>
            `,
            )
            .join("");

        const previewRows = profile.preview || [];
        const previewHeader = previewRows[0]
            ? Object.keys(previewRows[0]).map((key) => `<th>${key}</th>`).join("")
            : "";
        const previewBody = previewRows.length
            ? previewRows
                  .map((row, index) => {
                      const cells = Object.values(row)
                          .map((value) => `<td>${value ?? ""}</td>`)
                          .join("");
                      return `<tr><td>${index + 1}</td>${cells}</tr>`;
                  })
                  .join("")
            : "<tr><td colspan='99'>暂无数据</td></tr>";

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
                <tbody>${columnRows}</tbody>
            </table>
            <h4>数据预览</h4>
            <table>
                <thead><tr><th>#</th>${previewHeader}</tr></thead>
                <tbody>${previewBody}</tbody>
            </table>
        `;
    };

    const updateUploadPreview = (rows) => {
        if (!Array.isArray(rows) || !rows.length) {
            uploadPreview.hidden = true;
            uploadPreviewHead.innerHTML = "";
            uploadPreviewBody.innerHTML = "";
            return;
        }
        const columns = Object.keys(rows[0]);
        uploadPreviewHead.innerHTML = `<tr><th>#</th>${columns
            .map((c) => `<th>${c}</th>`)
            .join("")}</tr>`;
        const bodyHtml = rows
            .map((row, index) => {
                const cells = columns
                    .map((column) => `<td>${row[column] ?? ""}</td>`)
                    .join("");
                return `<tr><td>${index + 1}</td>${cells}</tr>`;
            })
            .join("");
        uploadPreviewBody.innerHTML = bodyHtml;
        uploadPreview.hidden = false;
    };

    const updateTrainDatasetLabel = () => {
        trainDatasetLabel.textContent = state.selectedDataset || "尚未选择";
    };

    const ensureSelectionDefaults = (datasetName, profile) => {
        const selection = getSelection(datasetName);
        if (selection.features.size === 0 && profile) {
            profile.column_details.forEach((column) => selection.features.add(column.name));
        }
        if (!selection.options) {
            selection.options = { ...defaultOptions };
        }
        return selection;
    };

    const applyOptionsToForm = (options) => {
        numericImputation.value = options.numeric_imputation || defaultOptions.numeric_imputation;
        categoricalImputation.value = options.categorical_imputation || defaultOptions.categorical_imputation;
        numericScaling.value = options.numeric_scaling || defaultOptions.numeric_scaling;
        categoricalEncoding.value = options.categorical_encoding || defaultOptions.categorical_encoding;
        constantFillValue.value = options.constant_fill_value || "";
        toggleConstantField();
    };

    const buildColumnSelectors = (profile) => {
        const selection = ensureSelectionDefaults(profile.name, profile);
        const savedTarget = selection.target;
        const featureSet = selection.features;

        targetColumnContainer.innerHTML = "";
        featureColumnContainer.innerHTML = "";

        profile.column_details.forEach((column) => {
            const targetId = `target-${column.name}`;
            const featureId = `feature-${column.name}`;

            const targetItem = document.createElement("label");
            targetItem.className = "column-item";
            targetItem.innerHTML = `
                <input type="radio" name="target-column" value="${column.name}" id="${targetId}">
                <span>${column.name}</span>
                <small>${column.dtype}</small>
            `;
            const targetInput = targetItem.querySelector("input");
            targetInput.checked = column.name === savedTarget;
            targetInput.addEventListener("change", () => {
                const currentSelection = getSelection(profile.name);
                currentSelection.target = column.name;
                if (currentSelection.features.has(column.name)) {
                    currentSelection.features.delete(column.name);
                }
                updateFeatureCheckboxStates(profile.name);
                resetPreview();
            });
            targetColumnContainer.appendChild(targetItem);

            const featureItem = document.createElement("label");
            featureItem.className = "column-item";
            featureItem.innerHTML = `
                <input type="checkbox" value="${column.name}" id="${featureId}">
                <span>${column.name}</span>
                <small>缺失 ${column.missing}</small>
            `;
            const featureInput = featureItem.querySelector("input");
            featureInput.checked = featureSet.has(column.name) && column.name !== savedTarget;
            featureInput.disabled = column.name === selection.target;
            featureInput.addEventListener("change", () => {
                const currentSelection = getSelection(profile.name);
                if (featureInput.checked) {
                    currentSelection.features.add(column.name);
                } else {
                    currentSelection.features.delete(column.name);
                }
                resetPreview();
            });
            featureColumnContainer.appendChild(featureItem);
        });
        updateFeatureCheckboxStates(profile.name);
    };

    const updateFeatureCheckboxStates = (datasetName) => {
        const selection = getSelection(datasetName);
        const featureInputs = featureColumnContainer.querySelectorAll("input[type='checkbox']");
        featureInputs.forEach((input) => {
            input.disabled = input.value === selection.target;
            if (input.disabled) {
                input.checked = false;
                selection.features.delete(input.value);
            }
        });
    };

    const updateOptionsFromInputs = (datasetName) => {
        const selection = getSelection(datasetName);
        selection.options = {
            numeric_imputation: numericImputation.value,
            categorical_imputation: categoricalImputation.value,
            numeric_scaling: numericScaling.value,
            categorical_encoding: categoricalEncoding.value,
            constant_fill_value: constantFillValue.value.trim(),
        };
    };

    const loadDatasets = async () => {
        datasetTableBody.innerHTML = "<tr><td colspan='5'>加载中...</td></tr>";
        state.datasets.clear();
        const response = await fetch("/api/datasets");
        const data = await safeJson(response);
        if (!response.ok) {
            datasetTableBody.innerHTML = `<tr><td colspan='5'>加载失败：${data.detail || "未知错误"}</td></tr>`;
            return;
        }
        const datasets = data.datasets || [];
        datasetTableBody.innerHTML = "";
        featureDatasetSelect.innerHTML = '<option value="" disabled selected>请选择数据集</option>';

        if (!datasets.length) {
            datasetTableBody.innerHTML = "<tr><td colspan='5'>暂未上传数据集</td></tr>";
            datasetDetail.hidden = true;
            targetColumnContainer.innerHTML = "";
            featureColumnContainer.innerHTML = "";
            state.selectedDataset = "";
            updateTrainDatasetLabel();
            return;
        }

        datasets.forEach((dataset) => {
            state.datasets.set(dataset.name, dataset);
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${dataset.name}</td>
                <td>${dataset.rows}</td>
                <td>${dataset.columns}</td>
                <td>${Array.isArray(dataset.target_suggestions) ? dataset.target_suggestions.join(", ") : "-"}</td>
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
            featureDatasetSelect.appendChild(option);
        });

        if (state.selectedDataset && state.datasets.has(state.selectedDataset)) {
            featureDatasetSelect.value = state.selectedDataset;
            showDatasetDetail(state.selectedDataset);
        } else {
            featureDatasetSelect.value = "";
            datasetDetail.hidden = true;
            datasetDetailContent.innerHTML = "";
            targetColumnContainer.innerHTML = "";
            featureColumnContainer.innerHTML = "";
            state.selectedDataset = "";
            updateTrainDatasetLabel();
        }
    };

    const showDatasetDetail = async (name) => {
        state.selectedDataset = name;
        updateTrainDatasetLabel();
        featureDatasetSelect.value = name;
        featureResponse.textContent = "";
        resetPreview();
        const selection = getSelection(name);

        if (state.datasetProfiles.has(name)) {
            const profile = state.datasetProfiles.get(name);
            datasetDetail.hidden = false;
            datasetDetailContent.innerHTML = renderDatasetProfile(profile);
            buildColumnSelectors(profile);
            applyOptionsToForm(selection.options);
            return;
        }

        datasetDetail.hidden = false;
        datasetDetailContent.innerHTML = "加载中...";
        try {
            const response = await fetch(`/api/datasets/${encodeURIComponent(name)}`);
            const data = await safeJson(response);
            if (!response.ok) {
                datasetDetailContent.textContent = data.detail || "加载失败";
                return;
            }
            state.datasetProfiles.set(name, data);
            datasetDetailContent.innerHTML = renderDatasetProfile(data);
            buildColumnSelectors(data);
            applyOptionsToForm(selection.options);
        } catch (error) {
            datasetDetailContent.textContent = `加载失败：${error}`;
        }
    };

    const deleteDataset = async (name) => {
        if (!window.confirm(`确定要删除数据集 ${name} 吗？`)) {
            return;
        }
        try {
            const response = await fetch(`/api/datasets/${encodeURIComponent(name)}`, {
                method: "DELETE",
            });
            if (response.ok) {
                uploadResponse.textContent = `数据集 ${name} 已删除。`;
                state.datasets.delete(name);
                state.datasetProfiles.delete(name);
                state.selections.delete(name);
                if (state.selectedDataset === name) {
                    state.selectedDataset = "";
                }
                await loadDatasets();
            } else {
                const data = await safeJson(response);
                uploadResponse.textContent = `删除失败：${data.detail || "未知错误"}`;
            }
        } catch (error) {
            uploadResponse.textContent = `删除失败：${error}`;
        }
    };

    const ensureConstantValue = (selection) => {
        const options = selection.options;
        const requiresConstant =
            options.numeric_imputation === "constant" ||
            options.categorical_imputation === "constant";
        if (requiresConstant && !options.constant_fill_value) {
            featureResponse.textContent = "请填写常量填充值后再继续。";
            return false;
        }
        return true;
    };

    const loadModels = async () => {
        modelTableBody.innerHTML = "<tr><td colspan='6'>加载中...</td></tr>";
        const response = await fetch("/api/models");
        const data = await safeJson(response);
        if (!response.ok) {
            modelTableBody.innerHTML = `<tr><td colspan='6'>加载失败：${data.detail || "未知错误"}</td></tr>`;
            return;
        }

        const models = data.models || [];
        modelTableBody.innerHTML = "";
        predictModelSelect.innerHTML = '<option value="" disabled selected>请选择模型</option>';
        shapModelSelect.innerHTML = '<option value="" disabled selected>请选择模型</option>';

        if (!models.length) {
            modelTableBody.innerHTML = "<tr><td colspan='6'>暂未训练模型</td></tr>";
            modelDetail.hidden = true;
        }

        models.forEach((model) => {
            const row = document.createElement("tr");
            const metrics = model.metrics || {};
            const primary = formatPrimaryMetric(metrics, model.mode);
            row.innerHTML = `
                <td>${model.name}</td>
                <td>${model.mode || "classification"}</td>
                <td>${model.algorithm || "-"}</td>
                <td>${primary}</td>
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

        if (state.lastModel) {
            const hasModel = [...predictModelSelect.options].some((opt) => opt.value === state.lastModel);
            predictModelSelect.value = hasModel ? state.lastModel : "";
            shapModelSelect.value = hasModel ? state.lastModel : "";
        }
    };

    const formatPrimaryMetric = (metrics, mode) => {
        if (!metrics || typeof metrics !== "object") {
            return "-";
        }
        if (mode === "regression") {
            if (typeof metrics.rmse === "number") {
                return `RMSE ${metrics.rmse.toFixed(4)}`;
            }
            if (typeof metrics.mae === "number") {
                return `MAE ${metrics.mae.toFixed(4)}`;
            }
        } else {
            if (typeof metrics.accuracy === "number") {
                return `准确率 ${(metrics.accuracy * 100).toFixed(2)}%`;
            }
            if (typeof metrics.f1_weighted === "number") {
                return `F1 ${(metrics.f1_weighted * 100).toFixed(2)}%`;
            }
        }
        const firstKey = Object.keys(metrics)[0];
        if (!firstKey) {
            return "-";
        }
        const value = metrics[firstKey];
        if (typeof value === "number") {
            return `${firstKey} ${value.toFixed(4)}`;
        }
        return `${firstKey}: ${value}`;
    };

    const showModelDetail = async (name) => {
        modelDetail.hidden = false;
        modelDetailContent.innerHTML = "加载中...";
        try {
            const response = await fetch(`/api/models/${encodeURIComponent(name)}`);
            const data = await safeJson(response);
            if (!response.ok) {
                modelDetailContent.textContent = data.detail || "加载失败";
                return;
            }
            modelDetailContent.innerHTML = renderModelDetail(data);
        } catch (error) {
            modelDetailContent.textContent = `加载失败：${error}`;
        }
    };

    const renderModelDetail = (model) => {
        const metrics = model.metrics || {};
        const mode = model.mode || "classification";
        const metricRows = Object.entries(metrics)
            .map(([key, value]) => {
                const formatted = typeof value === "number" ? value.toFixed(4) : value;
                return `<tr><td>${key}</td><td>${formatted}</td></tr>`;
            })
            .join("");

        let confusionSection = "";
        if (mode === "classification" && model.report?.confusion_matrix) {
            const labels = model.report.labels || [];
            const header = labels.map((label) => `<th>${label}</th>`).join("");
            const rows = model.report.confusion_matrix
                .map((row, index) => {
                    const cells = row.map((value) => `<td>${value}</td>`).join("");
                    const label = labels[index] ?? index;
                    return `<tr><th>${label}</th>${cells}</tr>`;
                })
                .join("");
            confusionSection = `
                <h4>混淆矩阵</h4>
                <table>
                    <thead><tr><th></th>${header}</tr></thead>
                    <tbody>${rows}</tbody>
                </table>
                <h4>分类报告</h4>
                <pre>${JSON.stringify(model.report.classification_report || {}, null, 2)}</pre>
            `;
        } else if (mode === "classification") {
            confusionSection = "<p class='muted'>暂无混淆矩阵数据。</p>";
        } else {
            confusionSection = "<p class='muted'>回归任务不提供混淆矩阵。</p>";
        }

        return `
            <p><strong>${model.name}</strong> · ${mode} · ${model.algorithm}</p>
            <p>训练耗时：${(model.training_time_seconds ?? 0).toFixed(2)} 秒 ｜ 数据集分层：${model.stratified_split ? "是" : "否"}</p>
            <h4>核心指标</h4>
            <table><tbody>${metricRows || "<tr><td colspan='2'>暂无指标</td></tr>"}</tbody></table>
            ${confusionSection}
            <p>特征列：${Array.isArray(model.feature_names) ? model.feature_names.join(", ") : "-"}</p>
        `;
    };

    const deleteModel = async (name) => {
        if (!window.confirm(`确定要删除模型 ${name} 吗？`)) {
            return;
        }
        try {
            const response = await fetch(`/api/models/${encodeURIComponent(name)}`, {
                method: "DELETE",
            });
            if (response.ok) {
                trainResponse.textContent = `模型 ${name} 已删除。`;
                if (state.lastModel === name) {
                    state.lastModel = "";
                }
                await loadModels();
            } else {
                const data = await safeJson(response);
                trainResponse.textContent = `删除失败：${data.detail || "未知错误"}`;
            }
        } catch (error) {
            trainResponse.textContent = `删除失败：${error}`;
        }
    };

    const buildMetricsSummary = (metrics, mode) => {
        if (!metrics || typeof metrics !== "object") {
            return "暂无指标";
        }
        if (mode === "regression") {
            const keys = ["rmse", "mae", "r2"];
            const items = keys
                .filter((key) => key in metrics)
                .map((key) => {
                    const value = metrics[key];
                    const formatted = typeof value === "number" ? value.toFixed(4) : value;
                    return `<li>${key}: ${formatted}</li>`;
                })
                .join("");
            return `<ul>${items || "<li>暂无指标</li>"}</ul>`;
        }
        const keys = ["accuracy", "f1_weighted", "balanced_accuracy"];
        const items = keys
            .filter((key) => key in metrics)
            .map((key) => {
                const value = metrics[key];
                const formatted = typeof value === "number" ? value.toFixed(4) : value;
                return `<li>${key}: ${formatted}</li>`;
            })
            .join("");
        return `<ul>${items || "<li>暂无指标</li>"}</ul>`;
    };

    const resetBackendState = async () => {
        try {
            await fetch("/api/reset", { method: "POST" });
        } catch (error) {
            console.warn("reset failed", error);
        }
    };

    // ------------------------------------------------------------------
    // Event bindings
    // ------------------------------------------------------------------
    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const fileInput = document.getElementById("dataset-file");
        if (!fileInput.files.length) {
            uploadResponse.textContent = "请先选择一个文件。";
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
            const data = await safeJson(response);
            if (response.ok) {
                uploadResponse.textContent = `上传成功：${data.dataset_name}`;
                updateUploadPreview(data.preview || []);
                fileInput.value = "";
                state.selectedDataset = data.dataset_name;
                state.datasetProfiles.delete(data.dataset_name);
                await loadDatasets();
            } else {
                uploadResponse.textContent = `上传失败：${data.detail || "未知错误"}`;
                uploadPreview.hidden = true;
            }
        } catch (error) {
            uploadResponse.textContent = `上传失败：${error}`;
            uploadPreview.hidden = true;
        }
    });

    refreshDatasetsBtn.addEventListener("click", loadDatasets);

    featureDatasetSelect.addEventListener("change", (event) => {
        const datasetName = event.target.value;
        if (!datasetName) {
            return;
        }
        showDatasetDetail(datasetName);
    });

    const optionInputs = [
        numericImputation,
        categoricalImputation,
        numericScaling,
        categoricalEncoding,
        constantFillValue,
    ];
    optionInputs.forEach((input) => {
        input.addEventListener("change", () => {
            if (!state.selectedDataset) {
                return;
            }
            toggleConstantField();
            updateOptionsFromInputs(state.selectedDataset);
            resetPreview();
        });
    });

    previewButton.addEventListener("click", async () => {
        if (!state.selectedDataset) {
            featureResponse.textContent = "请先选择数据集。";
            return;
        }
        updateOptionsFromInputs(state.selectedDataset);
        const selection = getSelection(state.selectedDataset);
        if (!ensureConstantValue(selection)) {
            return;
        }

        try {
            const payload = {
                dataset_name: state.selectedDataset,
                feature_columns: Array.from(selection.features),
                feature_engineering: selection.options,
                sample_size: 10,
            };
            if (!payload.feature_columns.length) {
                featureResponse.textContent = "请至少勾选一个特征列。";
                return;
            }
            featureResponse.textContent = "生成中...";
            const response = await fetch("/api/preview", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await safeJson(response);
            if (response.ok) {
                renderPreviewTable(previewTableHead, previewTableBody, data.feature_names || [], data.rows || []);
                previewTable.hidden = false;
                featureResponse.textContent = "预处理预览已更新。";
            } else {
                featureResponse.textContent = `生成失败：${data.detail || "未知错误"}`;
                resetPreview();
            }
        } catch (error) {
            featureResponse.textContent = `生成失败：${error}`;
            resetPreview();
        }
    });

    trainForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (!state.selectedDataset) {
            trainResponse.textContent = "请先在上方选择数据集并确认目标列/特征列。";
            return;
        }
        updateOptionsFromInputs(state.selectedDataset);
        const selection = getSelection(state.selectedDataset);
        if (!ensureConstantValue(selection)) {
            trainResponse.textContent = "请填写常量填充值后再训练。";
            return;
        }

        const features = Array.from(selection.features);
        if (!selection.target) {
            trainResponse.textContent = "请先选择目标列。";
            return;
        }
        if (!features.length) {
            trainResponse.textContent = "请至少选择一个特征列。";
            return;
        }

        const payload = {
            dataset_name: state.selectedDataset,
            target_column: selection.target,
            feature_columns: features,
            mode: trainModeSelect.value,
            algorithm: trainAlgorithmSelect.value,
            test_size: parseFloat(trainTestSizeInput.value),
            random_state: parseInt(trainRandomStateInput.value, 10),
            feature_engineering: selection.options,
            model_params: {},
        };

        if (trainModelParams.value.trim()) {
            try {
                payload.model_params = JSON.parse(trainModelParams.value);
            } catch (error) {
                trainResponse.textContent = "模型参数 JSON 格式不正确。";
                return;
            }
        }

        trainResponse.textContent = "训练中...";
        try {
            const response = await fetch("/api/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await safeJson(response);
            if (response.ok) {
                const metricsHtml = buildMetricsSummary(data.metrics, payload.mode);
                trainResponse.innerHTML = `<strong>训练完成：</strong>${metricsHtml}`;
                state.lastModel = data.model?.name || "";
                await loadModels();
            } else {
                trainResponse.textContent = `训练失败：${data.detail || "未知错误"}`;
            }
        } catch (error) {
            trainResponse.textContent = `训练失败：${error}`;
        }
    });

    refreshModelsBtn.addEventListener("click", loadModels);

    predictForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        predictResponse.textContent = "预测中...";
        let features;
        try {
            features = JSON.parse(predictFeatures.value || "{}");
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
            const data = await safeJson(response);
            if (response.ok) {
                const predictions = Array.isArray(data.predictions)
                    ? data.predictions.join(", ")
                    : data.predictions;
                const probabilityText = data.probabilities
                    ? `\n概率：${JSON.stringify(data.probabilities)}`
                    : "";
                predictResponse.textContent = `预测结果：${predictions}${probabilityText}`;
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
            const data = await safeJson(response);
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

    // ------------------------------------------------------------------
    // Initialisation
    // ------------------------------------------------------------------
    (async () => {
        await resetBackendState();
        toggleConstantField();
        await loadDatasets();
        await loadModels();
    })();
});
