const numberFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
const percentFormat = new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 });
const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'] };
const unlearningPageSize = 10;
const selectedTrainingIndices = new Set();
let trainingRowCount = 0;
let trainingRowOffset = 0;
let trainingPageRows = [];
let unlearningTaskType = 'binary_classification';
const plotLayout = {
    autosize: true,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'DM Sans, sans-serif', color: '#9c9bb6', size: 10 },
    margin: { l: 45, r: 16, t: 12, b: 42 },
    xaxis: { gridcolor: 'rgba(209,211,255,.07)', zerolinecolor: 'rgba(209,211,255,.1)', linecolor: 'rgba(209,211,255,.1)', tickfont: { size: 9 } },
    yaxis: { gridcolor: 'rgba(209,211,255,.07)', zerolinecolor: 'rgba(209,211,255,.1)', linecolor: 'rgba(209,211,255,.1)', tickfont: { size: 9 } },
    showlegend: true,
    legend: { orientation: 'h', x: 0, y: 1.12, font: { size: 9 } },
    hoverlabel: { bgcolor: '#202138', bordercolor: '#454461', font: { color: '#f0efff', size: 10 } }
};

function byId(id) { return document.getElementById(id); }

async function getJson(url, options = {}) {
    let response;
    try {
        response = await fetch(url, options);
    } catch (error) {
        error.apiRequest = true;
        throw error;
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        const error = new Error(payload.error || `Request failed (${response.status})`);
        error.requestId = payload.request_id;
        error.apiRequest = true;
        throw error;
    }
    return payload;
}

async function setActiveTarget(targetColumn) {
    return getJson('/api/target', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_column: targetColumn })
    });
}

function showToast(message) {
    const toast = byId('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('visible');
    window.clearTimeout(showToast.timeoutId);
    showToast.timeoutId = window.setTimeout(() => toast.classList.remove('visible'), 3200);
}

function setLoading(container, message = 'Loading live data') {
    if (!container) return;
    container.innerHTML = `<div class="loading-state"><span class="loader"></span><strong>${message}</strong></div>`;
}

function setEmpty(container, title = 'No data available', description = 'Run the relevant workflow to generate real results.') {
    if (!container) return;
    if (window.Plotly && container.data) window.Plotly.purge(container);
    container.innerHTML = '';
    const wrapper = document.createElement('div');
    wrapper.className = 'chart-empty';
    const heading = document.createElement('strong');
    heading.textContent = title;
    const detail = document.createElement('span');
    detail.textContent = description;
    wrapper.append(heading, detail);
    container.append(wrapper);
}

function setMetric(id, value, formatter = numberFormat) {
    const element = byId(id);
    if (!element) return;
    element.classList.remove('skeleton');
    const formatted = typeof formatter === 'function' ? formatter(Number(value)) : formatter.format(Number(value));
    element.textContent = value === null || value === undefined || !Number.isFinite(Number(value)) ? '—' : formatted;
}

const metricLabels = {
    accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1: 'F1 score', roc_auc: 'ROC AUC',
    mae: 'MAE', mse: 'MSE', rmse: 'RMSE', r2: 'R² score'
};

function renderMetricCards(containerId, metrics, taskType) {
    const container = byId(containerId);
    if (!container) return;
    container.replaceChildren();
    const keys = taskType === 'regression'
        ? ['mae', 'mse', 'rmse', 'r2']
        : ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'];
    keys.filter(key => key !== 'roc_auc' || metrics?.[key] !== null && metrics?.[key] !== undefined).forEach(key => {
        const card = document.createElement('div');
        card.className = 'result-metric';
        const label = document.createElement('span');
        label.textContent = metricLabels[key];
        const value = document.createElement('strong');
        const metricValue = metrics?.[key];
        value.textContent = metricValue === null || metricValue === undefined
            ? '—'
            : taskType === 'regression' ? numberFormat.format(metricValue) : percentFormat.format(metricValue);
        card.append(label, value);
        container.append(card);
    });
}

function setEstimatorOptions(selectId, supportedModels, selectedModel) {
    const select = byId(selectId);
    if (!select) return;
    select.replaceChildren();
    (supportedModels || []).forEach(model => {
        const option = document.createElement('option');
        option.value = model.key;
        option.textContent = model.name;
        option.selected = model.key === selectedModel;
        select.append(option);
    });
}

function defaultEstimator(taskType) {
    return taskType === 'regression' ? 'linear_regression' : 'logistic_regression';
}

function unlearningEstimator(taskType) {
    if (byId('unlearning-method')?.value === 'approximate') {
        return taskType === 'regression' ? 'sgd_regressor' : 'sgd_classifier';
    }
    return defaultEstimator(taskType);
}

function taskLabel(taskType) {
    if (taskType === 'regression') return 'Regression';
    if (taskType === 'binary_classification') return 'Binary classification';
    if (taskType === 'multiclass_classification') return 'Multiclass classification';
    return 'Unsupported target';
}

function renderPlot(id, traces, layout = {}, emptyMessage) {
    const element = byId(id);
    if (!element) return;
    if (!window.Plotly) {
        setEmpty(element, 'Charts unavailable', 'Plotly did not load. Check the browser network or CDN access and refresh.');
        return;
    }
    const hasPoints = traces.some(trace => Array.isArray(trace.x) ? trace.x.length > 0 : Array.isArray(trace.values) ? trace.values.length > 0 : Array.isArray(trace.z) ? trace.z.length > 0 : false);
    if (!hasPoints) {
        setEmpty(element, emptyMessage || 'No results yet', 'This chart will populate from the active dataset and completed model runs.');
        return;
    }
    if (element.data) window.Plotly.purge(element);
    element.innerHTML = '';
    const showRenderError = error => {
            console.error(`Unable to render chart "${id}":`, error);
            setEmpty(element, 'Chart rendering failed', error.message || 'Check the chart data and browser console.');
    };
    try {
        Promise.resolve(window.Plotly.newPlot(element, traces, { ...plotLayout, ...layout }, plotConfig))
            .catch(showRenderError);
    } catch (error) {
        showRenderError(error);
    }
}

function setChartTitle(slotId, title) {
    const heading = byId(slotId)?.closest('.chart-card')?.querySelector('.chart-heading h2');
    if (heading) heading.textContent = title;
}

function renderDashboardCharts(data) {
    const charts = data.charts || {};
    const methods = charts.method_comparison || {};
    const taskType = data.task_type || data.stats?.task_type || data.dataset?.task_type;
    const isRegression = taskType === 'regression';
    const metricIndex = isRegression ? 3 : 0;
    const primaryMetric = isRegression ? 'r2' : 'accuracy';
    const primaryFormat = isRegression ? '.3f' : '.1%';
    const primaryLabel = isRegression ? 'R² score' : 'Accuracy';
    if (byId('kpi-accuracy')?.previousElementSibling) byId('kpi-accuracy').previousElementSibling.textContent = isRegression ? 'Model R²' : 'Model accuracy';
    setChartTitle('chart-accuracy', isRegression ? 'Model R² comparison' : 'Model accuracy');
    setChartTitle('chart-metrics', isRegression ? 'Regression metrics' : 'Precision · Recall · F1');
    setChartTitle('chart-roc', isRegression ? 'Actual vs predicted' : 'ROC curve');
    setChartTitle('chart-pr', isRegression ? 'Residuals' : 'Precision–recall curve');
    byId('chart-confusion')?.closest('.chart-card')?.toggleAttribute('hidden', isRegression);
    const accuracyTraces = Object.entries(methods).filter(([, values]) => values && values[metricIndex] !== null && values[metricIndex] !== undefined).map(([name, values], index) => ({
        x: [name], y: [values[metricIndex]], type: 'bar', name, marker: { color: ['#a78bfa', '#67e8d1', '#f28bb8'][index % 3] },
        hovertemplate: `%{x}<br>${primaryLabel} %{y:${primaryFormat}}<extra></extra>`
    }));
    renderPlot('chart-accuracy', accuracyTraces, { yaxis: { ...plotLayout.yaxis, tickformat: isRegression ? '.3f' : '.0%', range: isRegression ? undefined : [0, 1] }, showlegend: false }, `Train a model to view ${primaryLabel}`);

    const comparison = charts.comparison_metrics || {};
    const metricTraces = [];
    if (comparison.original) metricTraces.push({ x: comparison.labels, y: comparison.original, type: 'bar', name: 'Original', marker: { color: '#a78bfa' } });
    if (comparison.unlearned) metricTraces.push({ x: comparison.labels, y: comparison.unlearned, type: 'bar', name: 'Latest unlearned', marker: { color: '#67e8d1' } });
    renderPlot('chart-metrics', metricTraces, { barmode: 'group', yaxis: { ...plotLayout.yaxis, range: isRegression ? undefined : [0, 1], tickformat: isRegression ? undefined : '.0%' } }, 'Train a model to compare scores');

    const confusion = charts.confusion_matrix || {};
    renderPlot('chart-confusion', confusion.values && confusion.values.length ? [{
        x: confusion.labels, y: confusion.labels, z: confusion.values, type: 'heatmap', colorscale: [[0, '#17182b'], [.55, '#7058b2'], [1, '#67e8d1']], showscale: true,
        hovertemplate: 'Predicted %{x}<br>Actual %{y}<br>Count %{z}<extra></extra>'
    }] : [], { xaxis: { ...plotLayout.xaxis, title: 'Predicted' }, yaxis: { ...plotLayout.yaxis, title: 'Actual', autorange: 'reversed' } }, 'Train a model to view the confusion matrix');

    const roc = charts.roc;
    byId('auc-label')?.replaceChildren(document.createTextNode(isRegression
        ? `R² ${data.stats?.metrics?.r2 === null || data.stats?.metrics?.r2 === undefined ? '—' : Number(data.stats.metrics.r2).toFixed(3)}`
        : `AUC ${charts.auc === null || charts.auc === undefined ? '—' : Number(charts.auc).toFixed(3)}`));
    const actualPredicted = charts.actual_vs_predicted;
    renderPlot('chart-roc', isRegression && actualPredicted ? [
        { x: actualPredicted.actual, y: actualPredicted.predicted, type: 'scatter', mode: 'markers', name: 'Predictions', marker: { color: '#67e8d1', size: 4, opacity: .5 } },
        { x: [Math.min(...actualPredicted.actual), Math.max(...actualPredicted.actual)], y: [Math.min(...actualPredicted.actual), Math.max(...actualPredicted.actual)], type: 'scatter', mode: 'lines', name: 'Ideal fit', line: { color: '#f5c46b', dash: 'dot' } }
    ] : !isRegression && roc ? [
        { x: roc.false_positive_rate, y: roc.true_positive_rate, type: 'scatter', mode: 'lines', name: 'ROC', line: { color: '#a78bfa', width: 2.5 }, fill: 'tozeroy', fillcolor: 'rgba(167,139,250,.08)' },
        { x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines', name: 'Chance', line: { color: '#696983', dash: 'dot', width: 1 } }
    ] : [], { xaxis: { ...plotLayout.xaxis, title: isRegression ? 'Actual' : 'False positive rate', range: isRegression ? undefined : [0, 1] }, yaxis: { ...plotLayout.yaxis, title: isRegression ? 'Predicted' : 'True positive rate', range: isRegression ? undefined : [0, 1] } }, isRegression ? 'Train a regression model to view predictions' : 'Train a model to view ROC');

    const pr = charts.precision_recall;
    const residuals = charts.residuals;
    renderPlot('chart-pr', isRegression && residuals ? [{ x: residuals.predicted, y: residuals.residuals, type: 'scatter', mode: 'markers', name: 'Residuals', marker: { color: '#f28bb8', size: 4, opacity: .5 }, hovertemplate: 'Predicted %{x:.2f}<br>Residual %{y:.2f}<extra></extra>' }, { x: [Math.min(...residuals.predicted), Math.max(...residuals.predicted)], y: [0, 0], type: 'scatter', mode: 'lines', name: 'Zero residual', line: { color: '#f5c46b', dash: 'dot' } }] : !isRegression && pr ? [{ x: pr.recall, y: pr.precision, type: 'scatter', mode: 'lines', name: 'Precision–recall', line: { color: '#67e8d1', width: 2.5 }, fill: 'tozeroy', fillcolor: 'rgba(103,232,209,.07)' }] : [], { xaxis: { ...plotLayout.xaxis, title: isRegression ? 'Predicted' : 'Recall', range: isRegression ? undefined : [0, 1] }, yaxis: { ...plotLayout.yaxis, title: isRegression ? 'Residual' : 'Precision', range: isRegression ? undefined : [0, 1] } }, isRegression ? 'Train a regression model to view residuals' : 'Train a model to view precision–recall');

    const distribution = charts.class_distribution;
    const distributionTrace = distribution?.values?.length
        ? isRegression
            ? [{ x: distribution.labels, y: distribution.values, type: 'bar', marker: { color: '#67e8d1' }, hovertemplate: '%{x}<br>%{y:,} records<extra></extra>' }]
            : [{ labels: distribution.labels, values: distribution.values, type: 'pie', hole: .62, sort: false, marker: { colors: ['#a78bfa', '#67e8d1', '#f28bb8', '#f5c46b'] }, textinfo: 'label+percent', textfont: { size: 9 }, hovertemplate: '%{label}<br>%{value:,} records (%{percent})<extra></extra>' }]
        : [];
    renderPlot('chart-distribution', distributionTrace, { showlegend: true, legend: { orientation: 'v', x: .7, y: .5 } }, 'No active dataset');
    const missing = charts.missing_values || {};
    renderPlot('chart-missing', missing.values?.some(value => value > 0) ? [{ x: missing.values, y: missing.labels, type: 'bar', orientation: 'h', marker: { color: '#f5c46b' }, hovertemplate: '%{y}<br>%{x:,} missing<extra></extra>' }] : [], { showlegend: false, margin: { ...plotLayout.margin, l: 115 }, xaxis: { ...plotLayout.xaxis, title: 'Missing cells' }, yaxis: { ...plotLayout.yaxis, autorange: 'reversed' } }, 'No missing values detected');
    const correlation = charts.correlation || {};
    renderPlot('chart-correlation', correlation.values?.length ? [{ x: correlation.labels, y: correlation.labels, z: correlation.values, type: 'heatmap', zmin: -1, zmax: 1, colorscale: [[0, '#67e8d1'], [.5, '#17182b'], [1, '#a78bfa']], hovertemplate: '%{y} × %{x}<br>Correlation %{z:.2f}<extra></extra>' }] : [], { margin: { l: 105, r: 14, t: 10, b: 65 }, xaxis: { ...plotLayout.xaxis, tickangle: -35 }, yaxis: { ...plotLayout.yaxis, autorange: 'reversed' }, showlegend: false }, 'No numeric features available');
    const importance = charts.feature_importance || [];
    renderPlot('chart-importance', importance.length ? [{ x: importance.map(item => item.importance).reverse(), y: importance.map(item => item.feature).reverse(), type: 'bar', orientation: 'h', marker: { color: '#a78bfa' }, hovertemplate: '%{y}<br>Importance %{x:.4f}<extra></extra>' }] : [], { showlegend: false, margin: { ...plotLayout.margin, l: 150 }, xaxis: { ...plotLayout.xaxis, title: 'Importance' }, yaxis: { ...plotLayout.yaxis, autorange: 'reversed' } }, 'Train a model to rank features');

    const forgotten = charts.forgetting_records || {};
    renderPlot('chart-forgetting', data.has_unlearning && forgotten.values?.length ? [{ labels: forgotten.labels, values: forgotten.values, type: 'pie', hole: .64, marker: { colors: ['#f28bb8', '#524b7f'] }, textinfo: 'label+value', hovertemplate: '%{label}: %{value:,}<extra></extra>' }] : [], {}, 'Run unlearning to view forgotten and retained records');
    const methodTraces = Object.entries(methods).filter(([, values]) => values).map(([name, values], index) => ({ x: charts.comparison_metrics?.labels || [], y: values, type: 'bar', name, marker: { color: ['#a78bfa', '#67e8d1', '#f28bb8'][index % 3] } }));
    renderPlot('chart-methods', methodTraces, { barmode: 'group', yaxis: { ...plotLayout.yaxis, range: isRegression ? undefined : [0, 1], tickformat: isRegression ? undefined : '.0%' } }, 'Run training and unlearning to compare methods');
    const times = charts.timings || {};
    const validTimings = times.values?.map((value, index) => ({ label: times.labels[index], value })).filter(item => item.value !== null && item.value !== undefined) || [];
    renderPlot('chart-timings', validTimings.length ? [{ x: validTimings.map(item => item.label), y: validTimings.map(item => item.value), type: 'bar', marker: { color: ['#a78bfa', '#67e8d1'] }, hovertemplate: '%{x}<br>%{y:.2f}s<extra></extra>' }] : [], { showlegend: false, yaxis: { ...plotLayout.yaxis, title: 'Seconds' } }, 'Run training or unlearning to compare execution time');
}

async function loadOverview() {
    document.querySelectorAll('#dashboard-charts .chart-slot').forEach(slot => setLoading(slot));
    const data = await getJson('/api/dashboard');
    const stats = data.stats || {};
    const taskType = data.task_type || stats.task_type || data.dataset?.task_type;
    setMetric('kpi-records', stats.total_records);
    const primaryMetric = taskType === 'regression' ? stats.metrics?.r2 : stats.metrics?.accuracy;
    setMetric('kpi-accuracy', primaryMetric, taskType === 'regression' ? numberFormat : percentFormat);
    setMetric('kpi-forgotten', data.has_unlearning ? stats.forgotten_records : null);
    const elapsed = [stats.training_time, stats.unlearning_time].filter(value => value !== null && value !== undefined);
    setMetric('kpi-runtime', elapsed.length ? Math.max(...elapsed) : null, value => `${value.toFixed(2)}s`);
    if (byId('kpi-columns')) byId('kpi-columns').textContent = `${data.dataset?.shape?.[1] ?? '—'} columns in active dataset`;
    if (byId('kpi-model')) byId('kpi-model').textContent = data.has_model ? `${stats.model_name || 'Model'} · held-out score` : 'No trained model';
    if (byId('kpi-retained')) byId('kpi-retained').textContent = data.has_unlearning ? `${numberFormat.format(stats.retained_records)} retained` : 'No unlearning run';
    if (byId('kpi-runtime-detail')) byId('kpi-runtime-detail').textContent = `Training ${formatSeconds(stats.training_time)} · Unlearning ${formatSeconds(stats.unlearning_time)}`;
    if (byId('overview-subtitle')) byId('overview-subtitle').textContent = `${data.dataset?.name || 'No active dataset'} · ${data.dataset?.rows ? numberFormat.format(data.dataset.rows) : 'No'} records · ${data.dataset?.source || 'source unavailable'} · ${taskLabel(taskType)} · target ${data.dataset?.target_column || 'not selected'}`;
    if (byId('target-label')) byId('target-label').textContent = (data.dataset?.target_column || 'TARGET').toUpperCase();
    if (byId('data-source')) byId('data-source').textContent = 'Live data · /api/dashboard';
    if (byId('model-empty')) byId('model-empty').hidden = Boolean(data.has_model);
    renderDashboardCharts(data);
    return data;
}

function formatSeconds(value) { return value === null || value === undefined ? '—' : `${Number(value).toFixed(2)}s`; }

async function loadDatasetPage() {
    const [dataset, preview, catalog] = await Promise.all([getJson('/api/dataset'), getJson('/api/dataset/preview'), getJson('/api/datasets')]);
    const missingCount = Object.values(dataset.missing_values || {}).reduce((total, value) => total + Number(value || 0), 0);
    byId('dataset-caption').textContent = `${numberFormat.format(dataset.rows)} rows · ${dataset.columns.length} columns`;
    byId('dataset-rows').textContent = numberFormat.format(dataset.rows);
    byId('dataset-columns').textContent = numberFormat.format(dataset.columns.length);
    byId('dataset-target-current').textContent = dataset.target_column || '—';
    byId('dataset-missing').textContent = numberFormat.format(missingCount);
    byId('dataset-status').innerHTML = '<span class="status-dot"></span>Active';
    byId('dataset-name').textContent = dataset.name || 'Active dataset';
    byId('dataset-source').textContent = dataset.source || 'Source unavailable';
    byId('dataset-task').textContent = `Detected task: ${taskLabel(dataset.task_type)}`;
    const activeDatasetSelect = byId('active-dataset');
    activeDatasetSelect.replaceChildren();
    catalog.datasets.forEach(item => {
        const option = document.createElement('option');
        option.value = item.key;
        option.textContent = `${item.name} · ${numberFormat.format(item.rows)} rows`;
        option.selected = item.key === dataset.dataset_key;
        activeDatasetSelect.append(option);
    });
    const selects = [byId('target-column'), byId('model-target'), byId('unlearning-target')].filter(Boolean);
    selects.forEach(select => {
        select.replaceChildren();
        dataset.columns.forEach(column => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            option.selected = column === dataset.target_column;
            select.append(option);
        });
    });
    const header = byId('preview-head');
    const body = byId('preview-body');
    header.replaceChildren();
    body.replaceChildren();
    const headRow = document.createElement('tr');
    preview.columns.forEach(column => { const th = document.createElement('th'); th.textContent = column; headRow.append(th); });
    header.append(headRow);
    if (!preview.rows.length) {
        const row = document.createElement('tr');
        row.innerHTML = '<td class="table-empty" colspan="100">No rows to preview.</td>';
        body.append(row);
    } else {
        preview.rows.forEach(record => {
            const row = document.createElement('tr');
            preview.columns.forEach(column => { const cell = document.createElement('td'); cell.textContent = record[column] ?? '—'; row.append(cell); });
            body.append(row);
        });
    }
    byId('preview-count').textContent = `Showing ${preview.rows.length} of ${numberFormat.format(dataset.rows)} records`;
}

async function handleDatasetActions() {
    byId('active-dataset')?.addEventListener('change', async event => {
        const select = event.currentTarget;
        select.disabled = true;
        byId('dataset-message').textContent = 'Activating selected dataset…';
        try {
            await getJson('/api/dataset/activate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_key: select.value })
            });
            await loadDatasetPage();
            byId('dataset-message').textContent = 'Active dataset updated.';
        } catch (error) {
            byId('dataset-message').textContent = error.message;
            await loadDatasetPage();
        } finally { select.disabled = false; }
    });
    const form = byId('upload-form');
    if (form) form.addEventListener('submit', async event => {
        event.preventDefault();
        const file = byId('dataset-file').files[0];
        if (!file) return showToast('Choose a CSV file first.');
        const button = byId('upload-button');
        button.disabled = true;
        byId('dataset-message').textContent = 'Validating and importing CSV…';
        try {
            const body = new FormData();
            body.append('file', file);
            const result = await getJson('/api/dataset/upload', { method: 'POST', body });
            byId('dataset-message').textContent = `${numberFormat.format(result.dataset.rows)} rows imported.`;
            await loadDatasetPage();
            showToast('Dataset uploaded and activated.');
        } catch (error) { byId('dataset-message').textContent = error.message; }
        finally { button.disabled = false; }
    });
    byId('target-button')?.addEventListener('click', async () => {
        try {
            const target = byId('target-column').value;
            await setActiveTarget(target);
            await loadDatasetPage();
            showToast(`Target set to ${target}.`);
        } catch (error) { showToast(error.message); }
    });
}

async function loadModelExplorer() {
    const dataset = await getJson('/api/dataset');
    byId('model-dataset-label').textContent = `${dataset.name} · ${numberFormat.format(dataset.rows)} records · ${dataset.target_column}`;
    byId('model-task-type').textContent = `Detected task: ${taskLabel(dataset.task_type)}`;
    setEstimatorOptions('model-type', dataset.supported_models, defaultEstimator(dataset.task_type));
    const target = byId('model-target');
    if (target) {
        target.replaceChildren();
        dataset.columns.forEach(column => { const option = document.createElement('option'); option.value = column; option.textContent = column; option.selected = column === dataset.target_column; target.append(option); });
        target.addEventListener('change', async () => {
            try {
                const targetInfo = await setActiveTarget(target.value);
                byId('model-dataset-label').textContent = `${dataset.name} · ${numberFormat.format(dataset.rows)} records · ${target.value}`;
                byId('model-task-type').textContent = `Detected task: ${taskLabel(targetInfo.task_type)}`;
                setEstimatorOptions('model-type', targetInfo.supported_models, defaultEstimator(targetInfo.task_type));
                await renderLatestModel();
                byId('training-message').textContent = `Active target: ${target.value}`;
            } catch (error) {
                byId('training-message').textContent = error.message;
            }
        });
    }
    await renderLatestModel();
}

async function renderLatestModel() {
    const result = await getJson('/api/model-explorer');
    const hasModel = result.status === 'ok';
    const taskType = result.task_type || (await getJson('/api/dataset')).task_type;
    byId('model-badge').textContent = hasModel ? result.model_name : 'Not trained';
    byId('model-summary').textContent = hasModel ? `${taskLabel(taskType)} · fitted ${result.model_name} · fixed held-out evaluation` : 'No model results available.';
    renderMetricCards('model-metrics', hasModel ? result.metrics : {}, taskType);
    const importance = result.feature_importance || [];
    renderPlot('model-importance', importance.length ? [{ x: importance.map(item => item.importance).reverse(), y: importance.map(item => item.feature).reverse(), type: 'bar', orientation: 'h', marker: { color: '#a78bfa' } }] : [], { showlegend: false, margin: { ...plotLayout.margin, l: 155 }, xaxis: { ...plotLayout.xaxis, title: 'Importance' } }, 'Train a model to view feature importance');
    const actualPredicted = result.actual_vs_predicted;
    renderPlot('model-actual-predicted', actualPredicted ? [{ x: actualPredicted.actual, y: actualPredicted.predicted, type: 'scatter', mode: 'markers', name: 'Predictions', marker: { color: '#67e8d1', size: 4, opacity: .5 } }] : [], { xaxis: { ...plotLayout.xaxis, title: 'Actual' }, yaxis: { ...plotLayout.yaxis, title: 'Predicted' } }, 'Actual vs predicted applies to regression targets');
    const residuals = result.residuals;
    renderPlot('model-residuals', residuals ? [{ x: residuals.predicted, y: residuals.residuals, type: 'scatter', mode: 'markers', name: 'Residuals', marker: { color: '#f28bb8', size: 4, opacity: .5 } }] : [], { xaxis: { ...plotLayout.xaxis, title: 'Predicted' }, yaxis: { ...plotLayout.yaxis, title: 'Residual' } }, 'Residual plots apply to regression targets');
    return hasModel;
}

async function handleTraining() {
    byId('train-model')?.addEventListener('click', async event => {
        const button = event.currentTarget;
        button.disabled = true;
        button.textContent = 'Training…';
        byId('training-message').textContent = 'Fitting the pipeline and evaluating held-out records…';
        try {
            const result = await getJson('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_name: byId('model-type').value, target_column: byId('model-target').value }) });
            byId('training-message').textContent = `Completed in ${Number(result.training_time).toFixed(2)} seconds.`;
            await renderLatestModel();
            showToast('Model training completed.');
        } catch (error) { byId('training-message').textContent = error.message; }
        finally { button.disabled = false; button.innerHTML = 'Train model <span aria-hidden="true">↗</span>'; }
    });
}

async function loadUnlearningPage() {
    const dataset = await getJson('/api/dataset');
    unlearningTaskType = dataset.task_type;
    byId('unlearning-dataset-label').textContent = `${dataset.name} · ${numberFormat.format(dataset.rows)} records · ${taskLabel(dataset.task_type)}`;
    const target = byId('unlearning-target');
    target.replaceChildren();
    if (!dataset.rows || !dataset.columns.length) {
        const option = document.createElement('option');
        option.textContent = 'No active dataset';
        target.append(option);
        byId('training-record-body').innerHTML = '<tr><td class="table-empty" colspan="100">Upload or select a dataset before requesting unlearning.</td></tr>';
        byId('selection-count').textContent = 'No training rows available';
        byId('unlearning-message').textContent = 'No active dataset is available.';
        byId('run-unlearning').disabled = true;
        return;
    }
    dataset.columns.forEach(column => { const option = document.createElement('option'); option.value = column; option.textContent = column; option.selected = column === dataset.target_column; target.append(option); });
    setEstimatorOptions('unlearning-model', dataset.supported_models, unlearningEstimator(dataset.task_type));
    byId('unlearning-method').addEventListener('change', event => {
        if (event.currentTarget.value === 'approximate') {
            const sgdModel = unlearningTaskType === 'regression' ? 'sgd_regressor' : 'sgd_classifier';
            if ([...byId('unlearning-model').options].some(option => option.value === sgdModel)) byId('unlearning-model').value = sgdModel;
        }
    });
    selectedTrainingIndices.clear();
    trainingRowOffset = 0;
    try {
        await loadTrainingSelectionPage(true);
    } catch (error) {
        byId('unlearning-message').textContent = error.message;
        byId('run-unlearning').disabled = true;
        return;
    }
    const dashboard = await getJson('/api/dashboard?view=unlearning');
    if (dashboard.has_unlearning) renderUnlearningResult(dashboard);
    else renderPlot('unlearning-comparison', [], {}, 'Run an unlearning experiment to compare results');
    target.addEventListener('change', async () => {
        byId('unlearning-message').textContent = `Setting active target to ${target.value}…`;
        try {
            const targetInfo = await setActiveTarget(target.value);
            unlearningTaskType = targetInfo.task_type;
            setEstimatorOptions('unlearning-model', targetInfo.supported_models, unlearningEstimator(targetInfo.task_type));
            byId('unlearning-dataset-label').textContent = `${dataset.name} · ${numberFormat.format(dataset.rows)} records · ${taskLabel(targetInfo.task_type)}`;
            selectedTrainingIndices.clear();
            trainingRowOffset = 0;
            await loadTrainingSelectionPage(true);
            byId('run-unlearning').disabled = false;
            byId('unlearning-message').textContent = 'Target updated. Choose training rows to forget.';
        } catch (error) { byId('unlearning-message').textContent = error.message; }
    });
}

function updateSelectionSummary() {
    const countInput = byId('forget-count');
    countInput.max = String(trainingRowCount);
    byId('selection-count').textContent = `${selectedTrainingIndices.size} selected · ${numberFormat.format(trainingRowCount)} training rows available`;
    countInput.value = String(selectedTrainingIndices.size);
}

function renderTrainingSelectionRows(rows, columns) {
    const head = byId('training-record-head');
    const body = byId('training-record-body');
    head.replaceChildren();
    body.replaceChildren();
    const headerRow = document.createElement('tr');
    ['Select', 'Record', ...columns].forEach(label => {
        const cell = document.createElement('th');
        cell.textContent = label;
        headerRow.append(cell);
    });
    head.append(headerRow);
    if (!rows.length) {
        const emptyRow = document.createElement('tr');
        const emptyCell = document.createElement('td');
        emptyCell.className = 'table-empty';
        emptyCell.colSpan = columns.length + 2;
        emptyCell.textContent = 'No training records on this page.';
        emptyRow.append(emptyCell);
        body.append(emptyRow);
        return;
    }
    rows.forEach((record, rowPosition) => {
        const row = document.createElement('tr');
        const checkboxCell = document.createElement('td');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = selectedTrainingIndices.has(record.index);
        checkbox.setAttribute('aria-label', `Select training record ${record.index + 1}`);
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) selectedTrainingIndices.add(record.index);
            else selectedTrainingIndices.delete(record.index);
            updateSelectionSummary();
            validateForgetCount(false);
        });
        checkboxCell.append(checkbox);
        row.append(checkboxCell);
        const indexCell = document.createElement('td');
        indexCell.textContent = String(record.index + 1);
        row.append(indexCell);
        columns.forEach(column => {
            const cell = document.createElement('td');
            cell.textContent = record[column] ?? '—';
            row.append(cell);
        });
        row.dataset.position = String(rowPosition);
        body.append(row);
    });
}

async function loadTrainingSelectionPage(defaultSelection = false) {
    const selection = await getJson(`/api/unlearning/selection?offset=${trainingRowOffset}&limit=${unlearningPageSize}`);
    trainingRowCount = selection.training_count;
    trainingPageRows = selection.rows;
    if (defaultSelection && selectedTrainingIndices.size === 0) {
        selection.rows.slice(0, Math.min(2, selection.training_count)).forEach(row => selectedTrainingIndices.add(row.index));
    }
    renderTrainingSelectionRows(selection.rows, Object.keys(selection.rows[0] || {}).filter(column => column !== 'index'));
    const currentPage = Math.floor(trainingRowOffset / unlearningPageSize) + 1;
    const pageCount = Math.max(1, Math.ceil(trainingRowCount / unlearningPageSize));
    byId('selection-page').textContent = `Page ${currentPage} of ${pageCount} · ${numberFormat.format(selection.testing_count)} held-out rows excluded`;
    byId('selection-prev').disabled = trainingRowOffset === 0;
    byId('selection-next').disabled = trainingRowOffset + unlearningPageSize >= trainingRowCount;
    updateSelectionSummary();
}

function validateForgetCount(showValid = false) {
    const rawCount = byId('forget-count').value.trim();
    const count = Number(rawCount);
    const message = byId('unlearning-message');
    if (!rawCount || !Number.isInteger(count) || count <= 0) {
        message.textContent = 'Enter a whole number greater than zero.';
        return false;
    }
    if (count > trainingRowCount) {
        message.textContent = `This dataset has only ${trainingRowCount} training rows available.`;
        return false;
    }
    if (count !== selectedTrainingIndices.size) {
        message.textContent = `The count is ${count}, but ${selectedTrainingIndices.size} training rows are selected.`;
        return false;
    }
    if (showValid) message.textContent = `${count} selected training rows are ready.`;
    return true;
}

function renderUnlearningResult(data) {
    const metrics = data.charts?.method_comparison || {};
    const stats = data.stats || {};
    const history = data.unlearning_history || [];
    const latest = history[0] || {};
    const methodKey = stats.unlearning_method || latest.method;
    const method = methodKey === 'approximate' ? 'Approximate · experimental' : methodKey === 'exact' ? 'Exact retraining' : 'Unlearning';
    const isRegression = data.task_type === 'regression';
    const metricKey = isRegression ? 'rmse' : 'accuracy';
    const metricName = isRegression ? 'RMSE' : 'Accuracy';
    const original = stats.unlearning_original_metrics?.[metricKey]
        ?? latest.original_metrics?.[metricKey]
        ?? stats.metrics?.[metricKey];
    const updated = stats.unlearned_metrics?.[metricKey] ?? latest.post_metrics?.[metricKey];
    const formatPerformance = value => value === null || value === undefined || !Number.isFinite(Number(value))
        ? '—'
        : isRegression ? numberFormat.format(Number(value)) : percentFormat.format(Number(value));
    byId('unlearning-badge').textContent = method;
    byId('unlearning-summary').textContent = history.length
        ? `${history.length} saved experiment${history.length === 1 ? '' : 's'} for this dataset target. Approximate runs remain experimental.`
        : 'No saved unlearning experiment is available for this dataset target.';
    byId('unlearning-model-result').textContent = stats.unlearning_model_name || latest.model_name || stats.model_name || '—';
    byId('unlearning-target-result').textContent = data.dataset?.target_column || stats.target_column || latest.target_column || '—';
    byId('forgot-count-result').textContent = numberFormat.format(stats.forgotten_records ?? latest.forgotten_records ?? 0);
    byId('remaining-count-result').textContent = numberFormat.format(stats.remaining_training_records ?? latest.remaining_training_records ?? 0);
    byId('original-performance-label').textContent = `Original ${metricName}`;
    byId('post-performance-label').textContent = `Post-unlearning ${metricName}`;
    byId('original-performance-result').textContent = formatPerformance(original);
    byId('post-performance-result').textContent = formatPerformance(updated);
    byId('unlearning-delta-label').textContent = `${metricName} delta`;
    const delta = Number(updated) - Number(original);
    byId('accuracy-delta').textContent = Number.isFinite(delta)
        ? isRegression ? delta.toFixed(3) : `${(delta * 100).toFixed(2)} pp`
        : '—';
    byId('retraining-time-result').textContent = formatSeconds(stats.retraining_time ?? latest.retraining_time);
    byId('unlearning-time-result').textContent = formatSeconds(stats.unlearning_time ?? latest.execution_time);
    byId('method-result').textContent = method;
    const comparison = data.charts?.comparison_metrics || {};
    const labels = comparison.labels || [];
    const comparisonTraces = [];
    if (comparison.original) comparisonTraces.push({ x: labels, y: comparison.original, type: 'bar', name: 'Before', marker: { color: '#a78bfa' } });
    if (comparison.unlearned) comparisonTraces.push({ x: labels, y: comparison.unlearned, type: 'bar', name: 'After', marker: { color: '#67e8d1' } });
    renderPlot('unlearning-comparison', comparisonTraces, { barmode: 'group', yaxis: { ...plotLayout.yaxis, range: isRegression ? undefined : [0, 1], tickformat: isRegression ? undefined : '.0%' } }, 'Run unlearning to compare held-out performance');

    const runLabel = run => `${run.method} · ${new Date(run.created_at).toLocaleString()}`;
    const methodNames = [...new Set(history.map(run => run.method))];
    const timeTraces = methodNames.map((name, index) => {
        const runs = history.filter(run => run.method === name && run.execution_time !== null && run.execution_time !== undefined);
        return { x: runs.map(runLabel), y: runs.map(run => run.execution_time), type: 'bar', name: name === 'approximate' ? 'Approximate · experimental' : 'Exact retraining', marker: { color: ['#a78bfa', '#67e8d1'][index % 2] } };
    }).filter(trace => trace.x.length);
    renderPlot('unlearning-method-times', timeTraces, { barmode: 'group', yaxis: { ...plotLayout.yaxis, title: 'Seconds' } }, 'No saved execution times yet');

    const balanceRuns = history.filter(run => run.remaining_training_records !== null && run.remaining_training_records !== undefined);
    renderPlot('unlearning-record-balance', balanceRuns.length ? [
        { x: balanceRuns.map(runLabel), y: balanceRuns.map(run => run.forgotten_records), type: 'bar', name: 'Forgotten', marker: { color: '#f28bb8' } },
        { x: balanceRuns.map(runLabel), y: balanceRuns.map(run => run.remaining_training_records), type: 'bar', name: 'Remaining training', marker: { color: '#524b7f' } }
    ] : [], { barmode: 'group', yaxis: { ...plotLayout.yaxis, title: 'Records' } }, 'No saved record counts yet');
    const performanceTraces = Object.entries(metrics)
        .filter(([, values]) => values)
        .map(([name, values], index) => ({
            x: labels,
            y: values,
            type: 'bar',
            name,
            marker: { color: ['#a78bfa', '#67e8d1', '#f28bb8'][index % 3] }
        }));
    renderPlot('unlearning-method-performance', performanceTraces, { barmode: 'group', yaxis: { ...plotLayout.yaxis, range: isRegression ? undefined : [0, 1], tickformat: isRegression ? undefined : '.0%' } }, 'No saved method performance to compare');

    const list = byId('unlearning-history');
    if (list) {
        list.replaceChildren();
        if (!history.length) {
            setEmpty(list, 'No saved unlearning runs', 'Completed experiments for this dataset target will appear here.');
        } else {
            history.forEach(run => {
                const row = document.createElement('article');
                row.className = 'experiment-row';
                const name = document.createElement('div');
                name.className = 'experiment-name';
                name.textContent = `${run.model_name} · ${run.target_column}`;
                const methodLabel = document.createElement('span');
                methodLabel.className = 'pill';
                methodLabel.textContent = run.method === 'approximate' ? 'Approximate · experimental' : 'Exact retraining';
                const details = document.createElement('span');
                details.className = 'experiment-meta';
                details.textContent = `${numberFormat.format(run.forgotten_records)} forgotten · ${numberFormat.format(run.remaining_training_records ?? 0)} remaining`;
                const date = document.createElement('time');
                date.className = 'experiment-meta';
                date.dateTime = run.created_at;
                date.textContent = new Date(run.created_at).toLocaleString();
                row.append(name, methodLabel, details, date);
                list.append(row);
            });
        }
    }
}

async function handleUnlearning() {
    byId('forget-count')?.addEventListener('input', () => validateForgetCount(false));
    byId('select-visible')?.addEventListener('click', () => {
        trainingPageRows.forEach(row => selectedTrainingIndices.add(row.index));
        renderTrainingSelectionRows(trainingPageRows, Object.keys(trainingPageRows[0] || {}).filter(column => column !== 'index'));
        updateSelectionSummary();
        validateForgetCount(false);
    });
    byId('clear-selection')?.addEventListener('click', () => {
        selectedTrainingIndices.clear();
        renderTrainingSelectionRows(trainingPageRows, Object.keys(trainingPageRows[0] || {}).filter(column => column !== 'index'));
        updateSelectionSummary();
        validateForgetCount(false);
    });
    byId('selection-prev')?.addEventListener('click', async () => {
        trainingRowOffset = Math.max(0, trainingRowOffset - unlearningPageSize);
        try { await loadTrainingSelectionPage(); } catch (error) { byId('unlearning-message').textContent = error.message; }
    });
    byId('selection-next')?.addEventListener('click', async () => {
        trainingRowOffset += unlearningPageSize;
        try { await loadTrainingSelectionPage(); } catch (error) { byId('unlearning-message').textContent = error.message; }
    });
    byId('run-unlearning')?.addEventListener('click', async event => {
        const button = event.currentTarget;
        if (!validateForgetCount(false)) return;
        const forgetIndices = [...selectedTrainingIndices].sort((left, right) => left - right);
        const method = byId('unlearning-method').value;
        const controller = new AbortController();
        const timeoutId = window.setTimeout(() => controller.abort(), 120000);
        const pendingMessageId = window.setTimeout(() => {
            byId('unlearning-message').textContent = 'Still waiting for the backend. Model fitting is synchronous; the held-out test set remains unchanged.';
        }, 8000);
        button.disabled = true;
        button.textContent = 'Running…';
        byId('unlearning-message').textContent = method === 'exact'
            ? `Request sent. Exact retraining fits the baseline and retained training rows; ${numberFormat.format(trainingRowCount - selectedTrainingIndices.size)} training rows and the held-out set remain.`
            : `Request sent. Approximate SGD is running against the same fixed held-out test set.`;
        try {
            const result = await getJson('/api/unlearn', { method: 'POST', signal: controller.signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ method, forget_count: Number(byId('forget-count').value), forget_indices: forgetIndices, target_column: byId('unlearning-target').value, model_name: byId('unlearning-model').value }) });
            if (result.status !== 'ok') throw new Error('The unlearning request did not complete.');
            window.clearTimeout(pendingMessageId);
            byId('unlearning-message').textContent = `${forgetIndices.length} selected training rows processed in ${Number(result.unlearning_time).toFixed(2)} seconds.`;
            showToast('Unlearning experiment completed.');
            try {
                const refreshController = new AbortController();
                const refreshTimeoutId = window.setTimeout(() => refreshController.abort(), 30000);
                let dashboard;
                try { dashboard = await getJson('/api/dashboard?view=unlearning', { signal: refreshController.signal }); }
                finally { window.clearTimeout(refreshTimeoutId); }
                renderUnlearningResult(dashboard);
            } catch (refreshError) {
                byId('unlearning-message').textContent = `Unlearning completed; dashboard refresh failed: ${refreshError.message}`;
            }
        } catch (error) {
            const detail = error.name === 'AbortError'
                ? 'No response after 120 seconds. The server may still be finishing; check its logs before retrying.'
                : `${error.message}${error.requestId ? ` (request ${error.requestId})` : ''}`;
            byId('unlearning-message').textContent = `Unlearning failed: ${detail}`;
        } finally {
            window.clearTimeout(timeoutId);
            window.clearTimeout(pendingMessageId);
            button.disabled = false;
            button.textContent = 'Run unlearning';
        }
    });
}

async function loadEvaluation() {
    const data = await getJson('/api/dashboard');
    const taskType = data.task_type;
    const isRegression = taskType === 'regression';
    const originalMetrics = data.stats?.metrics || {};
    byId('evaluation-empty').hidden = Boolean(data.has_model);
    byId('evaluation-caption').textContent = data.has_model ? `${taskLabel(taskType)} · ${data.stats.model_name || 'Model'} · ${numberFormat.format(data.stats.testing_records)} held-out records` : 'Metrics appear after a model has been trained.';
    renderMetricCards('evaluation-metrics', data.has_model ? originalMetrics : {}, taskType);
    document.querySelector('#eval-confusion').closest('.chart-card').hidden = isRegression;
    document.querySelector('#eval-roc').closest('.chart-card').hidden = isRegression;
    document.querySelector('#eval-pr').closest('.chart-card').hidden = isRegression;
    document.querySelector('#eval-actual-predicted').closest('.chart-card').hidden = !isRegression;
    document.querySelector('#eval-residuals').closest('.chart-card').hidden = !isRegression;
    document.querySelector('#eval-comparison').previousElementSibling.querySelector('h2').textContent = isRegression ? 'Regression metric comparison' : 'Classification metric comparison';
    const confusion = data.charts?.confusion_matrix;
    renderPlot('eval-confusion', confusion?.values?.length ? [{ x: confusion.labels, y: confusion.labels, z: confusion.values, type: 'heatmap', colorscale: [[0, '#17182b'], [.55, '#7058b2'], [1, '#67e8d1']] }] : [], { yaxis: { ...plotLayout.yaxis, autorange: 'reversed' } }, 'Train a model to view the confusion matrix');
    const roc = data.charts?.roc;
    byId('eval-auc').textContent = `AUC ${data.charts?.auc === null || data.charts?.auc === undefined ? '—' : Number(data.charts.auc).toFixed(3)}`;
    renderPlot('eval-roc', roc ? [{ x: roc.false_positive_rate, y: roc.true_positive_rate, type: 'scatter', mode: 'lines', name: 'ROC', line: { color: '#a78bfa', width: 2 } }, { x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines', name: 'Chance', line: { color: '#696983', dash: 'dot' } }] : [], { xaxis: { ...plotLayout.xaxis, title: 'False positive rate', range: [0, 1] }, yaxis: { ...plotLayout.yaxis, title: 'True positive rate', range: [0, 1] } }, 'Train a model to view ROC');
    const pr = data.charts?.precision_recall;
    renderPlot('eval-pr', pr ? [{ x: pr.recall, y: pr.precision, type: 'scatter', mode: 'lines', name: 'Precision–recall', line: { color: '#67e8d1', width: 2 } }] : [], { xaxis: { ...plotLayout.xaxis, title: 'Recall', range: [0, 1] }, yaxis: { ...plotLayout.yaxis, title: 'Precision', range: [0, 1] } }, 'Train a model to view precision–recall');
    const metricData = data.charts?.method_comparison || {};
    const labels = data.charts?.comparison_metrics?.labels || [];
    const traces = Object.entries(metricData).filter(([, values]) => values).map(([name, values], index) => ({ x: labels, y: values, type: 'bar', name, marker: { color: ['#a78bfa', '#67e8d1', '#f28bb8'][index % 3] } }));
    renderPlot('eval-comparison', traces, { barmode: 'group', yaxis: { ...plotLayout.yaxis, range: isRegression ? undefined : [0, 1], tickformat: isRegression ? undefined : '.0%' } }, 'Train a model to compare methods');
    const actualPredicted = data.charts?.actual_vs_predicted;
    renderPlot('eval-actual-predicted', actualPredicted ? [{ x: actualPredicted.actual, y: actualPredicted.predicted, type: 'scatter', mode: 'markers', name: 'Predictions', marker: { color: '#67e8d1', size: 4, opacity: .5 } }] : [], { xaxis: { ...plotLayout.xaxis, title: 'Actual' }, yaxis: { ...plotLayout.yaxis, title: 'Predicted' } }, 'Train a regression model to view predictions');
    const residuals = data.charts?.residuals;
    renderPlot('eval-residuals', residuals ? [{ x: residuals.predicted, y: residuals.residuals, type: 'scatter', mode: 'markers', name: 'Residuals', marker: { color: '#f28bb8', size: 4, opacity: .5 } }] : [], { xaxis: { ...plotLayout.xaxis, title: 'Predicted' }, yaxis: { ...plotLayout.yaxis, title: 'Residual' } }, 'Train a regression model to view residuals');
}

async function loadHistory() {
    const data = await getJson('/api/history');
    const list = byId('history-list');
    list.replaceChildren();
    const records = data.history || [];
    byId('history-count').textContent = `${records.length} ${records.length === 1 ? 'record' : 'records'}`;
    if (!records.length) {
        const empty = document.createElement('div');
        empty.className = 'chart-empty';
        empty.innerHTML = '<strong>No experiments recorded</strong><span>Train a model or run unlearning to create history.</span>';
        list.append(empty);
        return;
    }
    records.forEach(record => {
        const row = document.createElement('article');
        row.className = 'experiment-row';
        const name = document.createElement('div'); name.className = 'experiment-name'; name.textContent = record.name;
        const status = document.createElement('span'); status.className = 'pill'; status.textContent = record.status;
        const model = document.createElement('span'); model.className = 'experiment-meta';
        const selectedRows = (record.selected_indices || []).map(index => index + 1);
        model.textContent = record.method ? `${record.method} · ${record.forget_count} rows · records ${selectedRows.join(', ')}` : record.model_name || '—';
        if (record.method) model.title = `Selected dataset row indices: ${(record.selected_indices || []).join(', ')}`;
        const date = document.createElement('time'); date.className = 'experiment-meta'; date.dateTime = record.created_at; date.textContent = new Date(record.created_at).toLocaleString();
        row.append(name, status, model, date);
        list.append(row);
    });
}

async function loadReports() {
    const data = await getJson('/api/dashboard');
    const available = Boolean(data.has_model);
    byId('report-status').textContent = available ? `Latest model: ${data.stats.model_name} · accuracy ${percentFormat.format(data.stats.original_model_accuracy)}` : 'No model report available. Train a model before exporting results.';
    byId('report-badge').textContent = available ? 'Ready' : 'No results';
    byId('export-json').setAttribute('aria-disabled', String(!available));
    byId('export-csv').setAttribute('aria-disabled', String(!available));
    byId('export-json').classList.toggle('is-disabled', !available);
    byId('export-csv').classList.toggle('is-disabled', !available);
}

function initializeNavigation() {
    byId('menu-toggle')?.addEventListener('click', () => byId('sidebar').classList.toggle('is-open'));
    document.addEventListener('click', event => {
        const sidebar = byId('sidebar');
        if (sidebar?.classList.contains('is-open') && !sidebar.contains(event.target) && !byId('menu-toggle')?.contains(event.target)) sidebar.classList.remove('is-open');
    });
    byId('refresh-page')?.addEventListener('click', () => window.location.reload());
    byId('history-refresh')?.addEventListener('click', () => loadHistory().catch(error => showToast(error.message)));
    byId('chart-scope')?.addEventListener('change', event => {
        document.querySelectorAll('#dashboard-charts .chart-card').forEach(card => { card.hidden = event.target.value !== 'all' && card.dataset.category !== event.target.value; });
        window.setTimeout(resizePlots, 40);
    });
}

function resizePlots() {
    if (!window.Plotly) return;
    document.querySelectorAll('.chart-slot').forEach(node => { if (node.data) window.Plotly.Plots.resize(node); });
}

document.addEventListener('DOMContentLoaded', async () => {
    initializeNavigation();
    window.addEventListener('resize', resizePlots);
    const page = document.body.dataset.page;
    try {
        if (page === 'dashboard') await loadOverview();
        if (page === 'datasets') { await loadDatasetPage(); await handleDatasetActions(); }
        if (page === 'models') { await loadModelExplorer(); await handleTraining(); }
        if (page === 'unlearning') { await loadUnlearningPage(); await handleUnlearning(); }
        if (page === 'evaluation') await loadEvaluation();
        if (page === 'history') await loadHistory();
        if (page === 'reports') await loadReports();
    } catch (error) {
        console.error('ForgetLab page initialization failed:', error);
        showToast(error.message || 'Unable to load live application data.');
        const source = byId('data-source');
        if (source) source.textContent = error.apiRequest ? 'API unavailable' : 'Dashboard rendering error';
    }
});
