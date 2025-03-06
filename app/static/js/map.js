let map;
let labelLayer;  // Add this global variable

function initMap() {
    const today = getCurrentDate();
    
    // Create label layer separately
    labelLayer = new ol.layer.Tile({
        source: new ol.source.XYZ({
            url: 'https://tiles.stadiamaps.com/tiles/stamen_terrain_labels/{z}/{x}/{y}.png',
            maxZoom: 20
        }),
        zIndex: 1,
        opacity: 0.8
    });

    map = new ol.Map({
        target: 'map',
        layers: [
            // Base satellite layer
            new ol.layer.Tile({
                source: new ol.source.TileWMS({
                    url: 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi',
                    params: {
                        'LAYERS': 'VIIRS_SNPP_CorrectedReflectance_TrueColor',
                        'FORMAT': 'image/jpeg',
                        'VERSION': '1.3.0',
                        'TIME': today
                    },
                    projection: 'EPSG:4326',
                    crossOrigin: 'anonymous'
                })
            }),
            labelLayer  // Add the label layer
        ],
        view: new ol.View({
            center: [0, 0],
            projection: 'EPSG:4326',
            zoom: 3,
            minZoom: 1,
            maxZoom: 20,
            extent: [-180, -90, 180, 90]
        })
    });

    // Add opacity and label controls
    addMapControls();

    // Update reference layer toggles
    const referenceLabels = {
        'Labels': 2,  // City labels layer
        'Borders': 1  // Political boundaries layer
    };

    // Add layer controls
    addLayerControls();
    // Add date slider
    addDateSlider();

    // Add animation button handler
    document.getElementById('generate-animation').addEventListener('click', generateSmoothAnimation);
}

// Add function to create hide button
function createHideButton(container, controlName) {
    const hideButton = document.createElement('button');
    hideButton.className = 'hide-button';
    hideButton.innerHTML = '×';
    hideButton.title = 'Hide ' + controlName;
    
    hideButton.addEventListener('click', () => {
        container.classList.add('hidden');
        updateRestoreButton();
    });
    
    container.appendChild(hideButton);
}

// Add restore button
function addRestoreButton() {
    const restoreButton = document.createElement('button');
    restoreButton.className = 'restore-controls';
    restoreButton.textContent = 'Show Hidden Controls';
    restoreButton.style.display = 'none';
    
    restoreButton.addEventListener('click', () => {
        document.querySelectorAll('.layer-select, .reference-controls, .date-slider-container').forEach(el => {
            el.classList.remove('hidden');
        });
        updateRestoreButton();
    });
    
    document.body.appendChild(restoreButton);
    return restoreButton;
}

// Update restore button visibility
function updateRestoreButton() {
    const restoreButton = document.querySelector('.restore-controls');
    const hasHiddenControls = document.querySelectorAll('.hidden').length > 0;
    restoreButton.style.display = hasHiddenControls ? 'block' : 'none';
}

function addLayerControls() {
    const controlsDiv = document.getElementById('controls');
    
    // Add satellite layer selector
    const layerSelect = document.createElement('select');
    layerSelect.id = 'layer-select';
    layerSelect.className = 'layer-select';
    
    Object.entries(availableLayers).forEach(([name, value]) => {
        const option = document.createElement('option');
        option.value = value;
        option.text = name;
        layerSelect.appendChild(option);
    });
    
    layerSelect.onchange = (e) => {
        const selectedLayer = availableLayers[e.target.options[e.target.selectedIndex].text];
        const layer = map.getLayers().getArray()[0];
        
        let params = {
            'LAYERS': selectedLayer.id,
            'VERSION': '1.3.0',
            'TIME': getCurrentDate()
        };

        // Layer-specific configurations
        if (selectedLayer.id.includes('precipitation')) {
            params = {
                ...params,
                'FORMAT': 'image/png',
                'TRANSPARENT': 'TRUE',
                'STYLES': 'precipitation_rgb',
                'COLORSCALERANGE': '0.1,8',
                'NUMCOLORBANDS': '16',
                'LOGSCALE': 'true'
            };
            layer.setOpacity(0.8);
        } else if (selectedLayer.id.includes('Thermal_Anomalies')) {
            params = {
                ...params,
                'FORMAT': 'image/png',
                'TRANSPARENT': 'TRUE'
            };
            layer.setOpacity(0.8);
        } else if (selectedLayer.id.includes('Snow_Cover')) {
            params = {
                ...params,
                'FORMAT': 'image/png',
                'TRANSPARENT': 'TRUE'
            };
            layer.setOpacity(0.8);
        } else {
            params = {
                ...params,
                'FORMAT': 'image/jpeg'
            };
            layer.setOpacity(1.0);
        }

        // Update the layer source
        layer.getSource().updateParams(params);
        layer.getSource().refresh();

        // Update the legend
        updateLegend(selectedLayer);
    };

    // Add reference layer toggles
    const referenceDiv = document.createElement('div');
    referenceDiv.className = 'reference-controls';
    
    const referenceLabels = {
        'Coastlines': 1,
        'Borders': 2,
        'States': 3,
        'Labels': 4
    };

    Object.entries(referenceLabels).forEach(([name, index]) => {
        const container = document.createElement('div');
        container.className = 'reference-toggle';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `toggle-${name.toLowerCase()}`;
        checkbox.checked = true;

        const label = document.createElement('label');
        label.htmlFor = checkbox.id;
        label.textContent = name;

        checkbox.onchange = (e) => {
            const layer = map.getLayers().getArray()[index];
            layer.setVisible(e.target.checked);
        };

        container.appendChild(checkbox);
        container.appendChild(label);
        referenceDiv.appendChild(container);
    });

    // Add hide buttons to controls
    createHideButton(layerSelect, 'Layer Selector');
    createHideButton(referenceDiv, 'Reference Controls');

    // Add controls in correct order
    controlsDiv.insertBefore(layerSelect, controlsDiv.firstChild);
    controlsDiv.appendChild(referenceDiv);
    
    // Add restore button
    addRestoreButton();
}

function updateWMSLayer(date) {
    const layer = map.getLayers().getArray()[0];  // Get the satellite layer
    layer.getSource().updateParams({
        'TIME': date
    });
}

// Update the available layers
const availableLayers = {
    'True Color (VIIRS)': {
        id: 'VIIRS_SNPP_CorrectedReflectance_TrueColor',
        legend: null  // True color doesn't need a legend
    },
    'Temperature (Day)': {
        id: 'MODIS_Terra_Land_Surface_Temp_Day',
        legend: {
            title: 'Temperature (°C)',
            colors: ['#000080', '#0000D9', '#4000FF', '#8000FF', '#0080FF', '#00FFFF', '#00FF80', '#80FF00', '#FFFF00', '#FFC000', '#FF8000', '#FF4000', '#FF0000'],
            values: ['-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50', '60', '70']
        }
    },
    'Temperature (Night)': 'MODIS_Terra_Land_Surface_Temp_Night',
    'Precipitation (Daily)': {
        id: 'TRMM_3B42_Daily_7_precipitation',
        legend: {
            title: 'Precipitation (mm/day)',
            colors: ['#FFFFFF', '#A4F5A9', '#45BC55', '#10A0E0', '#2020FF', '#FA00FA', '#FFE800'],
            values: ['0', '0.1', '1', '5', '10', '20', '50+']
        }
    },
    'Snow Cover': 'MODIS_Terra_Snow_Cover',
    'Fire Detection': 'MODIS_Terra_Thermal_Anomalies_Day',
    'Vegetation Index': 'MODIS_Terra_NDVI_8Day',
    'Cloud Cover': 'MODIS_Terra_Cloud_Top_Temp_Day',
    'Aerosol': 'MODIS_Terra_Aerosol',
    'Sea Surface Temperature': 'MODIS_Terra_Sea_Surface_Temp_Day',
    'Ice Cover': 'MODIS_Terra_Sea_Ice'
};

function addDateSlider() {
    const controlsDiv = document.getElementById('controls');
    
    // Create date slider container
    const dateSliderContainer = document.createElement('div');
    dateSliderContainer.className = 'date-slider-container';
    
    // Add hide button to date slider
    createHideButton(dateSliderContainer, 'Date Controls');
    
    // Create date display
    const dateDisplay = document.createElement('div');
    dateDisplay.className = 'date-display';
    
    // Create slider controls
    const sliderControls = document.createElement('div');
    sliderControls.className = 'slider-controls';
    
    // Add play/pause button
    const playButton = document.createElement('button');
    playButton.className = 'play-button';
    playButton.innerHTML = '▶';
    playButton.title = 'Play/Pause';
    
    // Add date slider
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'date-slider';
    
    // Calculate date range (past 5 years to today)
    const today = new Date();
    today.setHours(0, 0, 0, 0); // Normalize to start of day
    const past5Years = new Date(today);
    past5Years.setFullYear(today.getFullYear() - 5);
    past5Years.setHours(0, 0, 0, 0); // Normalize to start of day
    
    const dayInMs = 24 * 60 * 60 * 1000; // One day in milliseconds
    
    // Set slider attributes
    slider.min = past5Years.getTime();
    slider.max = today.getTime();
    slider.value = today.getTime();
    slider.step = dayInMs;
    
    // Update date display
    function updateDateDisplay(date) {
        const options = { 
            weekday: 'short', 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric' 
        };
        dateDisplay.textContent = date.toLocaleDateString(undefined, options);
    }
    
    // Function to update everything based on a date
    function updateAll(date) {
        const normalizedDate = new Date(date);
        normalizedDate.setHours(0, 0, 0, 0); // Normalize to start of day
        slider.value = normalizedDate.getTime();
        updateDateDisplay(normalizedDate);
        updateWMSLayer(normalizedDate.toISOString().split('T')[0]);
    }
    
    // Initialize date display
    updateDateDisplay(new Date(parseInt(slider.value)));
    
    // Add event listeners
    slider.addEventListener('input', (e) => {
        const date = new Date(parseInt(e.target.value));
        updateAll(date);
    });
    
    // Animation variables
    let isPlaying = false;
    let animationInterval;
    
    // Cleanup function
    function cleanup() {
        if (animationInterval) {
            clearInterval(animationInterval);
            animationInterval = null;
        }
        isPlaying = false;
        playButton.innerHTML = '▶';
    }
    
    // Play/Pause functionality
    playButton.addEventListener('click', () => {
        isPlaying = !isPlaying;
        playButton.innerHTML = isPlaying ? '⏸' : '▶';
        
        if (isPlaying) {
            animationInterval = setInterval(() => {
                const currentValue = parseInt(slider.value);
                const nextValue = currentValue + dayInMs;
                
                if (nextValue > parseInt(slider.max)) {
                    slider.value = slider.min;
                    updateAll(new Date(parseInt(slider.min)));
                } else {
                    updateAll(new Date(nextValue));
                }
            }, 1000); // Update every second
        } else {
            cleanup();
        }
    });
    
    // Cleanup on container removal
    dateSliderContainer.addEventListener('remove', cleanup);
    
    // Add step backward/forward buttons
    const stepBackward = document.createElement('button');
    stepBackward.className = 'step-button';
    stepBackward.innerHTML = '←';
    stepBackward.title = 'Previous Day';
    stepBackward.onclick = () => {
        cleanup(); // Stop animation if running
        const currentValue = parseInt(slider.value);
        if (currentValue > parseInt(slider.min)) {
            const newDate = new Date(currentValue - dayInMs);
            updateAll(newDate);
        }
    };
    
    const stepForward = document.createElement('button');
    stepForward.className = 'step-button';
    stepForward.innerHTML = '→';
    stepForward.title = 'Next Day';
    stepForward.onclick = () => {
        cleanup(); // Stop animation if running
        const currentValue = parseInt(slider.value);
        if (currentValue < parseInt(slider.max)) {
            const newDate = new Date(currentValue + dayInMs);
            updateAll(newDate);
        }
    };
    
    // Assemble the controls
    sliderControls.appendChild(stepBackward);
    sliderControls.appendChild(playButton);
    sliderControls.appendChild(stepForward);
    sliderControls.appendChild(slider);
    
    dateSliderContainer.appendChild(dateDisplay);
    dateSliderContainer.appendChild(sliderControls);
    
    // Insert after reference controls
    const referenceControls = controlsDiv.querySelector('.reference-controls');
    controlsDiv.insertBefore(dateSliderContainer, referenceControls.nextSibling);
}

async function generateSmoothAnimation() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    
    // Get the current view state
    const view = map.getView();
    const zoom = view.getZoom();
    const extent = view.calculateExtent();
    const size = map.getSize();
    
    // Convert extent to [minLon, minLat, maxLon, maxLat] format
    const bbox = ol.proj.transformExtent(extent, view.getProjection(), 'EPSG:4326');
    
    console.log('Generating animation with zoom level:', zoom);
    
    const loadingDiv = document.getElementById('loading');
    loadingDiv.style.display = 'block';
    loadingDiv.textContent = 'Generating smooth animation...';
    
    try {
        const response = await fetch('/generate-rife-animation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                bbox: bbox,
                start_date: startDate,
                end_date: endDate,
                zoom_level: zoom,
                viewport_width: size[0],
                viewport_height: size[1]
            })
        });
        
        if (!response.ok) throw new Error('Animation generation failed');
        
        const data = await response.json();
        
        // Display the video in modal
        const videoModal = document.getElementById('video-modal');
        const video = document.getElementById('interpolated-video');
        const source = video.querySelector('source');
        
        // Update video source and type
        source.src = data.video_url;
        source.type = 'video/avi';
        
        // Load and play video
        video.load();
        videoModal.style.display = 'block';
        
        // Force play after a short delay
        setTimeout(() => {
            const playPromise = video.play();
            if (playPromise !== undefined) {
                playPromise.catch(error => {
                    console.log("Auto-play was prevented:", error);
                });
            }
        }, 1000);
        
    } catch (error) {
        console.error('Failed to generate animation:', error);
        alert('Failed to generate animation');
    } finally {
        loadingDiv.style.display = 'none';
    }
}

// Add modal close functionality
document.querySelector('.close-button').addEventListener('click', () => {
    document.getElementById('video-modal').style.display = 'none';
    document.getElementById('interpolated-video').pause();
});

// Close modal when clicking outside
window.addEventListener('click', (event) => {
    const modal = document.getElementById('video-modal');
    if (event.target === modal) {
        modal.style.display = 'none';
        document.getElementById('interpolated-video').pause();
    }
});

// Debounce function to limit API calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Function to parse coordinates from input
function parseCoordinates(input) {
    const coordRegex = /^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)$/;
    if (coordRegex.test(input)) {
        const [lat, lon] = input.split(',').map(coord => parseFloat(coord.trim()));
        return [lon, lat];  // OpenLayers uses [lon, lat] order
    }
    return null;
}

// Function to search locations using Nominatim
async function searchLocation(query) {
    const coordinates = parseCoordinates(query);
    if (coordinates) {
        return [{
            display_name: `Coordinates: ${query}`,
            lon: coordinates[0],
            lat: coordinates[1],
            boundingbox: [
                coordinates[1] - 0.1,
                coordinates[1] + 0.1,
                coordinates[0] - 0.1,
                coordinates[0] + 0.1
            ]
        }];
    }

    try {
        const response = await fetch(
            `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5&addressdetails=1&bounded=1`
        );
        const results = await response.json();
        return results;
    } catch (error) {
        console.error('Search failed:', error);
        return [];
    }
}

// Function to fly to location
function flyToLocation(lon, lat, zoom = 10) {
    const view = map.getView();
    const duration = 2000;
    
    const center = [parseFloat(lon), parseFloat(lat)];
    
    // Ensure the center is within valid bounds
    const maxExtent = [-180, -90, 180, 90];
    const clampedCenter = [
        Math.max(maxExtent[0], Math.min(maxExtent[2], center[0])),
        Math.max(maxExtent[1], Math.min(maxExtent[3], center[1]))
    ];
    
    // Animate the view with higher zoom level for better label visibility
    view.animate({
        center: clampedCenter,
        zoom: zoom,
        duration: duration,
        easing: ol.easing.easeOut
    });
}

// Initialize search functionality
function initializeSearch() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');

    const handleSearch = debounce(async (query) => {
        if (query.length < 2) {
            searchResults.classList.add('hidden');
            return;
        }

        const results = await searchLocation(query);
        searchResults.innerHTML = '';
        
        if (results.length === 0) {
            searchResults.classList.add('hidden');
            return;
        }

        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'search-result-item';
            resultItem.textContent = result.display_name;
            resultItem.addEventListener('click', () => {
                // Calculate zoom based on bounding box size
                let zoom = 10;
                if (result.boundingbox) {
                    const [south, north, west, east] = result.boundingbox.map(parseFloat);
                    const latSpan = Math.abs(north - south);
                    const lonSpan = Math.abs(east - west);
                    const span = Math.max(latSpan, lonSpan);
                    
                    // Adjust zoom based on area size
                    if (span < 0.1) zoom = 12;
                    else if (span < 0.5) zoom = 11;
                    else if (span < 1) zoom = 10;
                    else if (span < 5) zoom = 8;
                    else if (span < 10) zoom = 7;
                    else zoom = 6;
                }
                
                flyToLocation(result.lon, result.lat, zoom);
                searchInput.value = result.display_name;
                searchResults.classList.add('hidden');
            });
            searchResults.appendChild(resultItem);
        });

        searchResults.classList.remove('hidden');
    }, 300);

    searchInput.addEventListener('input', (e) => handleSearch(e.target.value));
    
    // Clear search on escape key
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            searchInput.value = '';
            searchResults.classList.add('hidden');
        }
    });

    // Close search results when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.classList.add('hidden');
        }
    });

    // Handle enter key
    searchInput.addEventListener('keypress', async (e) => {
        if (e.key === 'Enter' && searchInput.value) {
            const results = await searchLocation(searchInput.value);
            if (results.length > 0) {
                flyToLocation(results[0].lon, results[0].lat);
                searchResults.classList.add('hidden');
            }
        }
    });
}

// Simplified initialization
window.onload = () => {
    initMap();
    initializeSearch();
    
    // Add restore controls button
    const restoreButton = document.createElement('button');
    restoreButton.className = 'restore-controls';
    restoreButton.textContent = 'Show Controls';
    document.body.appendChild(restoreButton);
    
    restoreButton.addEventListener('click', () => {
        const controls = document.getElementById('controls');
        controls.style.display = 'block';
        restoreButton.style.display = 'none';
    });
    
    // Set default dates
    const today = new Date();
    const oneWeekAgo = new Date(today);
    oneWeekAgo.setDate(today.getDate() - 7);
    
    document.getElementById('end-date').value = today.toISOString().split('T')[0];
    document.getElementById('start-date').value = oneWeekAgo.toISOString().split('T')[0];
};

// Helper function to get current date in YYYY-MM-DD format
function getCurrentDate() {
    const today = new Date();
    return today.toISOString().split('T')[0];
}

// Add this function to create and update the legend
function updateLegend(layerInfo) {
    // Remove existing legend if any
    const existingLegend = document.getElementById('layer-legend');
    if (existingLegend) {
        existingLegend.remove();
    }

    // If no legend data, return
    if (!layerInfo.legend) return;

    // Create legend container
    const legend = document.createElement('div');
    legend.id = 'layer-legend';
    legend.className = 'layer-legend';

    // Add title
    const title = document.createElement('div');
    title.className = 'legend-title';
    title.textContent = layerInfo.legend.title;
    legend.appendChild(title);

    // Create gradient or discrete colors
    const colorBox = document.createElement('div');
    colorBox.className = 'legend-colors';

    // Create color bars and labels
    for (let i = 0; i < layerInfo.legend.colors.length - 1; i++) {
        const colorBar = document.createElement('div');
        colorBar.className = 'legend-color-bar';
        colorBar.style.background = layerInfo.legend.colors[i];
        
        const label = document.createElement('div');
        label.className = 'legend-label';
        label.textContent = layerInfo.legend.values[i];
        
        colorBox.appendChild(colorBar);
        colorBox.appendChild(label);
    }

    legend.appendChild(colorBox);
    document.body.appendChild(legend);
}

// Update the controls function
function addMapControls() {
    const controlsDiv = document.getElementById('controls');
    
    // Create controls container
    const mapControlsDiv = document.createElement('div');
    mapControlsDiv.className = 'map-controls';

    // Add opacity control
    const opacityControl = createOpacityControl();
    
    // Add label toggle
    const labelControl = createLabelControl();

    mapControlsDiv.appendChild(opacityControl);
    mapControlsDiv.appendChild(labelControl);
    
    // Insert controls at the top
    controlsDiv.insertBefore(mapControlsDiv, controlsDiv.firstChild);
}

function createOpacityControl() {
    const opacityControl = document.createElement('div');
    opacityControl.className = 'opacity-control';
    
    const label = document.createElement('label');
    label.textContent = 'Layer Transparency:';
    
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = '1';
    slider.step = '0.1';
    slider.value = '1';
    slider.className = 'opacity-slider';
    
    // Update the opacity slider to affect the satellite layer
    slider.addEventListener('input', (e) => {
        const satelliteLayer = map.getLayers().getArray()[0];
        satelliteLayer.setOpacity(parseFloat(e.target.value));
    });
    
    opacityControl.appendChild(label);
    opacityControl.appendChild(slider);
    
    return opacityControl;
}

function createLabelControl() {
    const labelControl = document.createElement('div');
    labelControl.className = 'label-control';
    
    const label = document.createElement('label');
    label.className = 'label-toggle';
    
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = true;
    
    const span = document.createElement('span');
    span.textContent = 'Show Labels';
    
    // Add event listener for label toggle
    checkbox.addEventListener('change', (e) => {
        labelLayer.setVisible(e.target.checked);
    });
    
    label.appendChild(checkbox);
    label.appendChild(span);
    labelControl.appendChild(label);
    
    return labelControl;
} 