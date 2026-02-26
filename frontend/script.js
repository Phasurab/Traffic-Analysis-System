document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Upload & UI
    const videoInput = document.getElementById('videoInput');
    const videoMsg = document.getElementById('videoMsg');
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('resultsSection');
    const outputVideo = document.getElementById('outputVideo');
    const outputJsonDisplay = document.getElementById('outputJsonDisplay');
    const downloadVideoBtn = document.getElementById('downloadVideoBtn');
    const downloadJsonBtn = document.getElementById('downloadJsonBtn');

    // DOM Elements - Workspace
    const drawingWorkspace = document.getElementById('drawingWorkspace');
    const shapeNameInput = document.getElementById('shapeNameInput');
    const btnSaveShape = document.getElementById('btnSaveShape');
    const shapesList = document.getElementById('shapesList');
    const geojsonValidation = document.getElementById('geojsonValidation');

    // GeoJSON Export Elements
    const geojsonModal = document.getElementById('geojsonModal');
    const geojsonOutputText = document.getElementById('geojsonOutputText');
    const btnCloseGeo = document.getElementById('btnCloseGeo');
    const btnDownloadGeoJSON = document.getElementById('btnDownloadGeoJSON');

    // Drawing State variables
    let savedShapes = []; // Array of shape objects { type: 'area'/'line', name: 'string', leafletLayer: L.Layer }
    let currentDrawnLayer = null; // The layer currently drawn but not yet named/saved

    // Temporary unmounted video to extract frames
    const tempVideo = document.createElement('video');

    // --- LEAFLET INITIALIZATION ---
    // Initialize map using simple CRS (flat image coordinates)
    const map = L.map('map', {
        crs: L.CRS.Simple,
        minZoom: -1,
        zoomControl: true,
        attributionControl: false
    });

    let imageOverlay = null;

    // FeatureGroup is to store editable layers
    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    // Initialize Leaflet Draw Control
    const drawControl = new L.Control.Draw({
        edit: {
            featureGroup: drawnItems
        },
        draw: {
            polygon: {
                shapeOptions: {
                    color: '#10b981',
                    weight: 5,
                    opacity: 0.8,
                    fillColor: '#10b981',
                    fillOpacity: 0.2
                }
            },
            polyline: {
                shapeOptions: {
                    color: '#38bdf8',
                    weight: 5,
                    opacity: 0.8
                }
            },
            rectangle: false,
            circle: false,
            marker: false,
            circlemarker: false
        }
    });

    // --- 1. Video Upload & Frame Extraction ---
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            videoMsg.textContent = file.name;
            videoMsg.style.color = '#38bdf8';

            // Extract Frame
            const fileURL = URL.createObjectURL(file);
            tempVideo.src = fileURL;
            tempVideo.muted = true;
            tempVideo.play().then(() => {
                tempVideo.pause();
                tempVideo.currentTime = 0.5; // grab frame at 0.5s to avoid black screen
            });
        }
    });

    tempVideo.addEventListener('seeked', () => {
        // Create an invisible canvas to extract the image dataURL
        const canvas = document.createElement('canvas');
        canvas.width = tempVideo.videoWidth;
        canvas.height = tempVideo.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(tempVideo, 0, 0, canvas.width, canvas.height);

        const frameDataUrl = canvas.toDataURL('image/jpeg');

        // Define Image bounds for Leaflet Simple CRS
        // Typically [ [0,0], [height, width] ]
        const bounds = [[0, 0], [tempVideo.videoHeight, tempVideo.videoWidth]];

        // If overlay exists, remove it
        if (imageOverlay) {
            map.removeLayer(imageOverlay);
        }

        // Add Image Overlay to Map
        imageOverlay = L.imageOverlay(frameDataUrl, bounds).addTo(map);
        map.fitBounds(bounds);

        // Add Drawing Controls now that video is loaded
        map.addControl(drawControl);

        drawingWorkspace.classList.remove('hidden');

        // Bugfix for leaflet map hidden container size issue
        setTimeout(() => { map.invalidateSize(); }, 200);
    });

    // --- 2. Leaflet Draw Event Handlers ---
    map.on(L.Draw.Event.CREATED, function (event) {
        const layer = event.layer;
        const type = event.layerType; // 'polygon' or 'polyline'

        // We temporarily add it to the map so user can see it, but don't add to drawnItems yet
        // until they provide a name and hit save
        currentDrawnLayer = layer;
        currentDrawnLayer.customType = (type === 'polygon') ? 'area' : 'line';
        map.addLayer(currentDrawnLayer);

        shapeNameInput.focus();
        btnSaveShape.style.animation = "pulse 1.5s infinite";
    });

    // --- 3. UI Shape Naming & Management ---
    btnSaveShape.addEventListener('click', () => {
        if (!currentDrawnLayer) {
            return alert("Please draw an area (Polygon) or line (Polyline) on the map first using the drawing toolbar.");
        }

        const name = shapeNameInput.value.trim();
        if (!name) return alert("Please enter a name for this shape! (e.g. 'Kaset Intersection' or 'Line 1')");

        // Attach custom name property to leaflet layer
        currentDrawnLayer.customName = name;

        // Move from map to editable feature group
        map.removeLayer(currentDrawnLayer);
        drawnItems.addLayer(currentDrawnLayer);

        savedShapes.push({
            type: currentDrawnLayer.customType,
            name: name,
            layerId: drawnItems.getLayerId(currentDrawnLayer)
        });

        // Update UI list
        renderShapesList();

        if (savedShapes.length > 0) {
            geojsonValidation.classList.remove('hidden');
        }

        // Reset state
        currentDrawnLayer = null;
        shapeNameInput.value = "";
        btnSaveShape.style.animation = "";
    });

    shapesList.addEventListener('click', (e) => {
        if (e.target.classList.contains('delete-shape-btn')) {
            const index = e.target.getAttribute('data-index');
            const shapeObj = savedShapes[index];

            // Remove from leaflet map
            const layer = drawnItems.getLayer(shapeObj.layerId);
            if (layer) {
                drawnItems.removeLayer(layer);
            }

            // Remove from array
            savedShapes.splice(index, 1);

            // Re-render list
            renderShapesList();

            if (savedShapes.length === 0) {
                geojsonValidation.classList.add('hidden');
            }
        }
    });

    function renderShapesList() {
        shapesList.innerHTML = '';
        savedShapes.forEach((s, i) => {
            const li = document.createElement('li');
            li.innerHTML = `[${s.type.toUpperCase()}] ${s.name} <button class="delete-shape-btn" data-index="${i}">✕</button>`;
            shapesList.appendChild(li);
        });
    }

    // --- 4. Package to format_input.json & Submit ---
    // Helper to generate the exact spec required by backend
    function generateGeoJSONSpec() {
        const specJson = {
            "zones": { "type": "FeatureCollection", "features": [] },
            "lane_set": {}
        };

        // We iterate through drawnItems instead of savedShapes array directly to ensure 
        // we capture any post-creation edits made via Leaflet's edit tools!
        drawnItems.eachLayer(function (layer) {
            // Leaflet LatLng extraction bounds are inverted [Y, X] in simple CRS, 
            // but we need [X, Y] for our backend.

            const feature = {
                "type": "Feature",
                "properties": { "name": layer.customName },
                "geometry": {}
            };

            // Get coordinates from Leaflet layer
            const latLngs = layer.getLatLngs();

            // Leaflet's Simple CRS origin (0,0) is at the bottom-left, meaning Y increases upwards.
            // Image coordinates (and OpenCV) origin (0,0) is at the top-left, meaning Y increases downwards.
            // We MUST flip the Y axis: y = actualVideoHeight - lat
            const actualVideoHeight = tempVideo.videoHeight;

            if (layer.customType === 'area') {
                // Leaflet polygons are nested arrays
                const polyCoords = latLngs[0].map(ll => [
                    Math.round(ll.lng),
                    Math.round(actualVideoHeight - ll.lat)
                ]);
                // Ensure polygon is closed for valid GeoJSON
                polyCoords.push([...polyCoords[0]]);

                feature.geometry.type = "Polygon";
                feature.geometry.coordinates = [polyCoords];
            } else {
                // Lines
                const lineCoords = latLngs.map(ll => [
                    Math.round(ll.lng),
                    Math.round(actualVideoHeight - ll.lat)
                ]);
                feature.geometry.type = "LineString";
                feature.geometry.coordinates = lineCoords;
            }

            // Both areas and lines should be registered in the lane mapping
            specJson.lane_set[layer.customName] = [layer.customName];
            specJson.zones.features.push(feature);
        });

        return specJson;
    }

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Ensure video exist and shapes drawn
        const videoFile = videoInput.files[0];
        const modelType = document.getElementById('modelSelect').value;

        if (!videoFile) {
            alert('Please select a video file.');
            return;
        }

        if (savedShapes.length === 0) {
            alert('Please draw at least one Area or Line on the map to track vehicles before submitting.');
            return;
        }

        const specJson = generateGeoJSONSpec();

        // Convert spec object to File Blob
        const jsonBlob = new Blob([JSON.stringify(specJson)], { type: 'application/json' });

        // Prepare UI state
        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing...';
        loader.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        drawingWorkspace.classList.add('hidden');

        // Form data prep
        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('json_spec', jsonBlob, 'drawn_spec.json');
        formData.append('model_type', modelType);

        try {
            // Send API Request
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to queue files.');
            }

            const data = await response.json();
            const taskId = data.task_id;

            // Setup dynamic progress UI
            const loaderText = document.querySelector('.loader-text');
            const modelName = modelType === 'yolo' ? 'YOLOv8' : 'DINOv3 LTDETR';
            loaderText.textContent = `Processing video with ${modelName}... 0%`;

            // Poll for progress
            const maxRetries = 600; // 10 minutes max at 1s intervals
            let retries = 0;

            const pollInterval = setInterval(async () => {
                retries++;
                const statusRes = await fetch(`/api/progress/${taskId}`);
                const statusData = await statusRes.json();

                if (statusData.status === 'processing') {
                    loaderText.textContent = `Processing video with ${modelName}... ${statusData.progress}%`;
                    const innerBar = document.getElementById('progressInner');
                    if (innerBar) innerBar.style.width = `${statusData.progress}%`;
                } else if (statusData.status === 'complete') {
                    clearInterval(pollInterval);
                    loaderText.textContent = `Processing complete! Loading results...`;
                    const innerBar = document.getElementById('progressInner');
                    if (innerBar) innerBar.style.width = `100%`;

                    // Build Download URLs
                    const videoUrl = `/api/download/video/${statusData.output_video}`;
                    const jsonUrl = `/api/download/json/${statusData.output_json}`;
                    const csvUrl = `/api/download/csv/${statusData.output_csv}`;

                    // Fetch final JSON to display in the UI
                    const resJson = await fetch(jsonUrl);
                    const parsedJson = await resJson.json();

                    // Render Results
                    outputVideo.src = videoUrl;

                    // User Friendly HTML Formatting
                    renderFriendlyResults(parsedJson);

                    downloadVideoBtn.href = videoUrl;
                    downloadVideoBtn.download = statusData.output_video;
                    downloadVideoBtn.target = "_blank";

                    downloadJsonBtn.href = jsonUrl;
                    downloadJsonBtn.download = statusData.output_json;
                    downloadJsonBtn.target = "_blank";

                    const downloadCsvBtn = document.getElementById('downloadCsvBtn');
                    // Changed back to using the physical accumulated CSV generated by pipeline.py
                    downloadCsvBtn.href = `/api/download/csv/accumulated_traffic_data.csv`;
                    downloadCsvBtn.download = `accumulated_traffic_data.csv`;
                    downloadCsvBtn.target = "_blank";

                    // Show results
                    resultsSection.classList.remove('hidden');
                    resultsSection.scrollIntoView({ behavior: 'smooth' });

                    // Restore UI state
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Run Inference Pipeline';
                    loader.classList.add('hidden');
                } else if (statusData.status === 'error') {
                    clearInterval(pollInterval);
                    alert(`Error processing files: ${statusData.error}`);
                    // Restore UI state
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Run Inference Pipeline';
                    loader.classList.add('hidden');
                }

                if (retries > maxRetries) {
                    clearInterval(pollInterval);
                    alert("Processing timed out.");
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Run Inference Pipeline';
                    loader.classList.add('hidden');
                }
            }, 1000);

        } catch (error) {
            console.error(error);
            alert(`Error processing files: ${error.message}`);
            // Restore UI state
            submitBtn.disabled = false;
            submitBtn.textContent = 'Run Inference Pipeline';
            loader.classList.add('hidden');
        }
    });

    // --- 5. GeoJSON Export Functionality ---
    geojsonValidation.addEventListener('click', () => {
        geojsonModal.classList.remove('hidden');
        geojsonOutputText.textContent = JSON.stringify(generateGeoJSONSpec(), null, 2);
        // Scroll to bottom
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    });

    geojsonValidation.style.cursor = 'pointer';
    geojsonValidation.title = 'Click to view raw GeoJSON';

    btnCloseGeo.addEventListener('click', () => {
        geojsonModal.classList.add('hidden');
    });

    btnDownloadGeoJSON.addEventListener('click', () => {
        const specJson = generateGeoJSONSpec();
        const jsonBlob = new Blob([JSON.stringify(specJson, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(jsonBlob);

        const a = document.createElement('a');
        a.href = url;
        a.download = 'format_input.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

});

function renderFriendlyResults(data) {
    const container = document.getElementById('outputJsonDisplay');
    container.innerHTML = '';

    // Areas
    if (data.areas && Object.keys(data.areas).length > 0) {
        let areaHtml = '<h4 style="color: #38bdf8; margin-bottom: 10px;">Area / Traffic Counts</h4>';
        areaHtml += '<div style="background: rgba(15, 23, 42, 0.4); border-radius: 8px; overflow: hidden;"><table class="results-table"><thead><tr><th>Area/Lane</th><th>Vehicle Class</th><th>Count</th></tr></thead><tbody>';
        for (const [lane, records] of Object.entries(data.areas)) {
            if (records.length === 0) continue;
            for (const record of records) {
                const [vclass, count] = Object.entries(record)[0];
                areaHtml += `<tr><td>${lane}</td><td>${vclass}</td><td>${count}</td></tr>`;
            }
        }
        areaHtml += '</tbody></table></div>';
        container.innerHTML += areaHtml;
    }

    // Paths
    if (data.paths && Object.keys(data.paths).length > 0) {
        let pathHtml = '<h4 style="color: #a78bfa; margin-top: 20px; margin-bottom: 10px;">Traffic Trajectories (A to B)</h4>';
        pathHtml += '<div style="background: rgba(15, 23, 42, 0.4); border-radius: 8px; overflow: hidden;"><table class="results-table"><thead><tr><th>Trajectory Path</th><th>Vehicle Class</th><th>Count</th></tr></thead><tbody>';
        for (const [pathName, records] of Object.entries(data.paths)) {
            if (records.length === 0) continue;
            for (const record of records) {
                const [vclass, count] = Object.entries(record)[0];
                pathHtml += `<tr><td>${pathName}</td><td>${vclass}</td><td>${count}</td></tr>`;
            }
        }
        pathHtml += '</tbody></table></div>';
        container.innerHTML += pathHtml;
    }

    // Fallback if no data
    if (container.innerHTML === '') {
        container.innerHTML = '<p style="color: #94a3b8;">No vehicles detected in defined areas.</p>';
    }
}
