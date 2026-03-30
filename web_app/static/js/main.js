$(document).ready(function() {
    // 变量初始化
    let cameraActive = false;
    let videoFeedInterval;
    let currentLanguage = 'zh';
    const I18N = {
        zh: {
            brand: '实验室安全检测系统',
            nav: { analysis: '场景分析', live: '实时检测', upload: '上传检测', info: '系统信息', settings: '设置' },
            hero: { title: '实验室安全监测系统', subtitle: '实时目标检测与大模型行为验证，守护实验室安全', start: '开始实时检测', upload: '上传检测', info: '系统介绍' },
            headers: { analysis: '场景分析与安全要求设定', live: '实时摄像头检测', imageUpload: '图像上传检测', videoUpload: '视频上传检测', info: '系统信息', history: '检测历史记录' },
            modelinfo: { title: '模型信息', intro: '本系统基于 YOLOv8 模型构建，用于实验室现场安全行为检测与防护装备识别，支持实时摄像头流、单张图像及视频文件的检测。请将模型权重放置于 models/lab_safety_detection6/weights/ 下的 best.pt。', model_name: '模型名称:', num_classes: '类别数量:', input_size: '输入尺寸:', class_list: '检测类别:' },
            live_buttons: { start: '启动摄像头', stop: '停止摄像头', capture: '截图' },
            settings: { title: '检测与报警设置', conf_label: '置信度阈值', labels: '显示标签', alerts: '启用报警', cancel: '取消', save: '保存设置', language: '语言' },
            history: { title: '检测历史记录', refresh: '刷新', table: ['来源', '类别', '帧', '时间'] },
            analysis: { title: '场景分析 (VLM)', upload_label: '上传实验室场景图片', desc_label: '描述 (可选)', desc_placeholder: '例如：这是一个化学实验室，正在进行酸碱滴定实验...', submit: 'AI 自动识别' },
            requirements: { title: '安全检测要求', hint: '控制是否检测对应的违规项', drinking: '禁止饮水 (No Drinking)', eating: '禁止进食 (No Eating)', gloves: '必须戴手套 (Gloves)', goggles: '必须戴护目镜 (Goggles)', mask: '必须戴口罩 (Mask)', coat: '必须穿实验服 (Lab Coat)', head: '必须戴头套 (Head Mask)', update: '手动更新要求' },
            stats: { title: '实时统计', waiting: '等待数据...' },
            upload_buttons: { image: '上传并检测', video: '上传并检测' },
            upload_labels: { image_file: '选择图像文件', video_file: '选择视频文件' },
            upload_controls: { select: '选择文件', none: '未选择文件' },
            usage: ['实时检测：启动摄像头进行实时安全检测', '图像检测：上传实验室图像进行安全分析', '视频检测：上传视频文件进行帧分析', '支持检测实验室安全相关物品和违规行为'],
            demo_warning: { strong: '注意：', text: '本系统为演示版本，仅供参考，部署到生产环境前请校验模型与检测阈值，并做好隐私与安全评估。' },
            status: { ready: '系统就绪', model: '模型', alerts_on: '报警: 开', alerts_off: '报警: 关' }
        },
        en: {
            brand: 'Lab Safety Detection System',
            nav: { analysis: 'Analysis', live: 'Live', upload: 'Upload', info: 'Info', settings: 'Settings' },
            hero: { title: 'Lab Safety Monitoring System', subtitle: 'Real-time detection with VLM verification for safer labs', start: 'Start Live Detection', upload: 'Upload Detection', info: 'About' },
            headers: { analysis: 'Scene Analysis & Safety Requirements', live: 'Live Camera Detection', imageUpload: 'Image Upload Detection', videoUpload: 'Video Upload Detection', info: 'System Info', history: 'Detection History' },
            modelinfo: { title: 'Model Info', intro: 'Built on YOLOv8 for lab safety behavior and PPE detection; supports live camera, image and video analysis. Place best.pt under models/lab_safety_detection6/weights/.', model_name: 'Model Name:', num_classes: 'Num Classes:', input_size: 'Input Size:', class_list: 'Classes:' },
            live_buttons: { start: 'Start Camera', stop: 'Stop Camera', capture: 'Capture' },
            settings: { title: 'Detection & Alerts Settings', conf_label: 'Confidence Threshold', labels: 'Show Labels', alerts: 'Enable Alerts', cancel: 'Cancel', save: 'Save', language: 'Language' },
            history: { title: 'Detection History', refresh: 'Refresh', table: ['Source', 'Class', 'Frame', 'Time'] },
            analysis: { title: 'Scene Analysis (VLM)', upload_label: 'Upload Lab Scene Image', desc_label: 'Description (optional)', desc_placeholder: 'e.g., A chemistry lab doing acid-base titration...', submit: 'AI Analyze' },
            requirements: { title: 'Safety Detection Requirements', hint: 'Toggle checks for specific violations', drinking: 'No Drinking', eating: 'No Eating', gloves: 'Gloves Required', goggles: 'Goggles Required', mask: 'Mask Required', coat: 'Lab Coat Required', head: 'Head Mask Required', update: 'Update Requirements' },
            stats: { title: 'Real-time Statistics', waiting: 'Waiting for data...' },
            upload_buttons: { image: 'Upload & Detect', video: 'Upload & Detect' },
            upload_labels: { image_file: 'Select Image File', video_file: 'Select Video File' },
            upload_controls: { select: 'Select File', none: 'No file selected' },
            usage: ['Live: Start camera for real-time safety detection', 'Image: Upload lab image for safety analysis', 'Video: Upload video for frame-wise analysis', 'Supports PPE detection and unsafe behaviors'],
            demo_warning: { strong: 'Note:', text: 'Demo only. Validate models and thresholds before production deployment and ensure privacy & safety compliance.' },
            status: { ready: 'Ready', model: 'Model', alerts_on: 'Alerts: On', alerts_off: 'Alerts: Off' }
        }
    };

    function applyLanguage(lang) {
        currentLanguage = (lang === 'en') ? 'en' : 'zh';
        const t = I18N[currentLanguage];
        $('.navbar-brand').html('<i class="fas fa-flask"></i> ' + t.brand);
        $('a[href="#analysis"]').text(t.nav.analysis);
        $('a[href="#live"]').text(t.nav.live);
        $('a[href="#upload"]').text(t.nav.upload);
        $('a[href="#info"]').text(t.nav.info);
        $('a[data-bs-target="#settingsModal"]').text(t.nav.settings);
        $('.hero-title').text(t.hero.title);
        $('.hero-subtitle').text(t.hero.subtitle);
        $('#heroStart').html('<i class="fas fa-play"></i> ' + t.hero.start);
        $('#heroUpload').html('<i class="fas fa-upload"></i> ' + t.hero.upload);
        $('#analysis .card-header').html('<i class="fas fa-brain"></i> ' + t.headers.analysis);
        $('#live .card-header').html('<i class="fas fa-video"></i> ' + t.headers.live);
        const uploadHeaders = $('#upload .card-header');
        if (uploadHeaders.length >= 2) {
            $(uploadHeaders[0]).html('<i class="fas fa-image"></i> ' + t.headers.imageUpload);
            $(uploadHeaders[1]).html('<i class="fas fa-film"></i> ' + t.headers.videoUpload);
        }
        $('#info .card-header').html('<i class="fas fa-info-circle"></i> ' + t.headers.info);
        $('#modelInfoTitle').text(t.modelinfo.title);
        $('#modelInfoIntro').text(t.modelinfo.intro);
        $('#labelModelName').text(t.modelinfo.model_name);
        $('#labelNumClasses').text(t.modelinfo.num_classes);
        $('#labelInputSize').text(t.modelinfo.input_size);
        $('#labelClassList').text(t.modelinfo.class_list);
        $('#startCamera').html('<i class="fas fa-play"></i> ' + t.live_buttons.start);
        $('#stopCamera').html('<i class="fas fa-stop"></i> ' + t.live_buttons.stop);
        $('#captureFrame').html('<i class="fas fa-camera"></i> ' + t.live_buttons.capture);
        $('#settingsModal .modal-title').text(t.settings.title);
        $('label[for="settingsLanguage"]').text(t.settings.language);
        $('label[for="settingsConf"]').html(t.settings.conf_label + ': <span id="settingsConfValue">' + $('#settingsConfValue').text() + '</span>');
        $('label[for="settingsShowLabels"]').text(t.settings.labels);
        $('label[for="settingsAlerts"]').text(t.settings.alerts);
        $('#saveSettings').text(t.settings.save);
        $('#settingsModal .btn-secondary').text(t.settings.cancel);
        $('#historyCard .card-header').html('<i class="fas fa-history"></i> ' + t.history.title + ' <button id="refreshHistory" class="btn btn-sm btn-outline-secondary float-end">' + t.history.refresh + '</button>');
        $('#historyTable thead tr').html('<th>' + t.history.table[0] + '</th><th>' + t.history.table[1] + '</th><th>' + t.history.table[2] + '</th><th>' + t.history.table[3] + '</th>');
        $('#analysisTitle').text('1. ' + t.analysis.title);
        $('#analysisUploadLabel').text(t.analysis.upload_label);
        $('#analysisDescLabel').text(t.analysis.desc_label);
        $('#analysisDesc').attr('placeholder', t.analysis.desc_placeholder);
        $('#analysisSubmit').html('<i class="fas fa-magic"></i> ' + t.analysis.submit);
        $('#requirementsTitle').text('2. ' + t.requirements.title);
        $('#requirementsHint').text(t.requirements.hint);
        $('label[for="req_drinking"]').text(t.requirements.drinking);
        $('label[for="req_eating"]').text(t.requirements.eating);
        $('label[for="req_gloves"]').text(t.requirements.gloves);
        $('label[for="req_goggles"]').text(t.requirements.goggles);
        $('label[for="req_mask"]').text(t.requirements.mask);
        $('label[for="req_coat"]').text(t.requirements.coat);
        $('label[for="req_head"]').text(t.requirements.head);
        $('#updateRequirements').text(t.requirements.update);
        $('#statsTitle').html('<i class="fas fa-chart-bar"></i> ' + t.stats.title);
        $('#statsWaiting').text(t.stats.waiting);
        $('#imageUploadButton').html('<i class="fas fa-upload"></i> ' + t.upload_buttons.image);
        $('#labelImageFile').text(t.upload_labels.image_file);
        $('#imageSelectBtn').html('<i class="fas fa-folder-open"></i> ' + t.upload_controls.select);
        $('#selectedImageName').text(t.upload_controls.none);
        $('#videoUploadButton').html('<i class="fas fa-upload"></i> ' + t.upload_buttons.video);
        $('#labelVideoFile').text(t.upload_labels.video_file);
        $('#videoSelectBtn').html('<i class="fas fa-folder-open"></i> ' + t.upload_controls.select);
        $('#selectedVideoName').text(t.upload_controls.none);
        if (Array.isArray(t.usage)) {
            const ul = $('#usageList');
            ul.html('');
            t.usage.forEach(function(item) { ul.append('<li>' + item + '</li>'); });
        }
        $('#demoWarningStrong').text(t.demo_warning.strong);
        $('#demoWarningText').text(t.demo_warning.text);
        // 状态栏文本（保留模型名）
        const rawMsg = $('#statusMessage').text();
        let modelName = 'lab_safety_detection';
        const m1 = rawMsg.match(/模型:\s*(\S+)/);
        const m2 = rawMsg.match(/Model:\s*(\S+)/);
        if (m1 && m1[1]) modelName = m1[1];
        else if (m2 && m2[1]) modelName = m2[1];
        $('#statusMessage').text(t.status.ready + ' | ' + t.status.model + ': ' + modelName);
        // 报警按钮文字
        if ($('#toggleAlerts').hasClass('btn-danger')) {
            $('#toggleAlerts').text(t.status.alerts_on);
        } else {
            $('#toggleAlerts').text(t.status.alerts_off);
        }
    }
    
    // 首页封面按钮交互
    $('#heroStart').on('click', function() {
        const target = $('#live');
        $('html, body').animate({ scrollTop: target.offset().top - 60 }, 400);
        $('#startCamera').trigger('click');
    });
    $('#heroUpload').on('click', function() {
        const target = $('#upload');
        $('html, body').animate({ scrollTop: target.offset().top - 60 }, 400);
    });
    
    // 页面加载时获取模型信息
    getModelInfo();

    // 报警开关（默认开启），以及从服务器加载设置
    let alertsEnabled = true;
    function updateToggleButton() {
        if (alertsEnabled) {
            $('#toggleAlerts').removeClass('btn-secondary').addClass('btn-danger').text(I18N[currentLanguage].status.alerts_on);
        } else {
            $('#toggleAlerts').removeClass('btn-danger').addClass('btn-secondary').text(I18N[currentLanguage].status.alerts_off);
        }
    }
    $('#toggleAlerts').click(function() {
        alertsEnabled = !alertsEnabled;
        updateToggleButton();
        // Persist alerts setting to server
        $.ajax({ url: '/set_settings', method: 'POST', contentType: 'application/json', data: JSON.stringify({ alerts_enabled: alertsEnabled }) });
    });

    // 加载当前服务器设置
    function loadSettings() {
        $.ajax({ url: '/get_settings', method: 'GET', success: function(s) {
            $('#settingsConf').val(s.conf_threshold || 0.3);
            $('#settingsConfValue').text(s.conf_threshold || 0.3);
            $('#settingsShowLabels').prop('checked', !!s.show_labels);
            $('#settingsAlerts').prop('checked', !!s.alerts_enabled);
            $('#settingsLanguage').val(s.language || 'zh');
            alertsEnabled = !!s.alerts_enabled;
            updateToggleButton();
            applyLanguage(s.language || 'zh');
        }});
    }
    loadSettings();
    
    // 摄像头控制
    $('#startCamera').click(function() {
        if (!cameraActive) {
            startCamera();
        }
    });
    
    $('#stopCamera').click(function() {
        if (cameraActive) {
            stopCamera();
        }
    });
    
    // 截图功能
    $('#captureFrame').click(function() {
        captureFrame();
    });
    
    // 图像上传
    $('#imageUploadForm').submit(function(e) {
        e.preventDefault();
        uploadImage();
    });
    
    // 视频上传
    $('#videoUploadForm').submit(function(e) {
        e.preventDefault();
        uploadVideo();
    });
    
    // 设置模态置信度滑块
    $('#settingsConf').on('input', function() {
        $('#settingsConfValue').text($(this).val());
    });

    // 保存设置
    $('#saveSettings').click(function() {
        const conf = parseFloat($('#settingsConf').val());
        const showLabels = $('#settingsShowLabels').prop('checked');
        const alerts = $('#settingsAlerts').prop('checked');
        const language = $('#settingsLanguage').val();
        $.ajax({ url: '/set_settings', method: 'POST', contentType: 'application/json', data: JSON.stringify({
            conf_threshold: conf,
            show_labels: showLabels,
            alerts_enabled: alerts,
            language: language
        }), success: function(resp) {
            if (resp.status === 'ok') {
                alertsEnabled = !!resp.settings.alerts_enabled;
                updateToggleButton();
                const smEl = document.getElementById('settingsModal');
                const sm = bootstrap.Modal.getInstance(smEl) || new bootstrap.Modal(smEl);
                sm.hide();
                updateStatus('设置已保存', 'success');
                applyLanguage(resp.settings.language || 'zh');
            } else {
                updateStatus('保存设置失败', 'danger');
            }
        }, error: function() { updateStatus('保存设置失败', 'danger'); } });
    });
    
    // 获取模型信息
    function getModelInfo() {
        $.ajax({
            url: '/model_info',
            method: 'GET',
            success: function(data) {
                if (data.error) {
                    $('#modelInfoIntro').text('模型加载失败');
                    return;
                }
                $('#modelName').text('lab_safety_detection6');
                $('#numClasses').text(data.num_classes);
                $('#inputSize').text(data.input_size);
                const ul = $('#classList');
                ul.html('');
                for (const [id, name] of Object.entries(data.names)) {
                    ul.append(`<li class="list-group-item">${id}: ${name}</li>`);
                }
            },
            error: function() {
                $('#modelInfoIntro').text('无法获取模型信息');
            }
        });
    }
    
    // 启动摄像头
    function startCamera() {
        $.ajax({
            url: '/start_camera',
            method: 'GET',
            success: function() {
                cameraActive = true;
                $('#videoFeed').show();
                $('#noVideo').hide();
                $('#startCamera').prop('disabled', true);
                $('#stopCamera').prop('disabled', false);
                $('#captureFrame').prop('disabled', false);
                
                // 设置视频流
                $('#videoFeed').attr('src', '/video_feed?' + new Date().getTime());
                
                // 开始获取统计信息
                startStatsUpdate();
                
                updateStatus('摄像头已启动', 'success');
            },
            error: function() {
                updateStatus('摄像头启动失败', 'danger');
            }
        });
    }
    
    // 停止摄像头
    function stopCamera() {
        $.ajax({
            url: '/stop_camera',
            method: 'GET',
            success: function() {
                cameraActive = false;
                $('#videoFeed').hide();
                $('#noVideo').show();
                $('#startCamera').prop('disabled', false);
                $('#stopCamera').prop('disabled', true);
                $('#captureFrame').prop('disabled', true);
                
                // 停止获取统计信息
                if (videoFeedInterval) {
                    clearInterval(videoFeedInterval);
                }
                
                updateStatus('摄像头已停止', 'warning');
            },
            error: function() {
                updateStatus('摄像头停止失败', 'danger');
            }
        });
    }
    
    // 更新统计信息
    function startStatsUpdate() {
        // 先立即获取一次
        updateDetectionStats();

        // 然后每 200ms 更新一次以接近实时
        videoFeedInterval = setInterval(updateDetectionStats, 200);
    }
    
    function updateDetectionStats() {
        $.ajax({
            url: '/get_detection_stats',
            method: 'GET',
            success: function(data) {
                let html = `
                    <p><strong>FPS:</strong> ${data.fps || 0}</p>
                    <p><strong>总检测数(Session):</strong> ${Object.values(data.session_counts || {}).reduce((a, b) => a + b, 0)}</p>
                `;
                
                // 显示累计统计 (Session Counts)
                if (data.session_counts) {
                    html += '<hr><h6 class="small">累计识别次数:</h6><ul class="list-unstyled small">';
                    const orderedKeys = ['Drinking', 'Eating', 'Gloves', 'Googles', 'Head Mask', 'Lab Coat', 'Mask', 'No Gloves', 'No Head Mask', 'No Lab coat', 'No Mask', 'No googles'];
                    
                    orderedKeys.forEach(key => {
                         const count = data.session_counts[key] || 0;
                         // 高亮显示有计数的项 (白色加粗)，无计数的项 (半透明白色)
                         const style = count > 0 ? 'font-weight:bold; color:#fff;' : 'color:rgba(255,255,255,0.6);';
                         html += `<li style="${style}">${key}: ${count}</li>`;
                    });
                    
                    // Add any other keys not in the list (if any)
                    for (const [key, count] of Object.entries(data.session_counts)) {
                        if (!orderedKeys.includes(key) && count > 0) {
                             html += `<li>${key}: ${count}</li>`;
                        }
                    }
                    html += '</ul>';
                } else if (data.by_class && Object.keys(data.by_class).length > 0) {
                     // Fallback to current frame if session_counts not available
                    html += '<ul>';
                    for (const [className, count] of Object.entries(data.by_class)) {
                        html += `<li>${className}: ${count}</li>`;
                    }
                    html += '</ul>';
                } else {
                    html += '<p class="text-muted">未检测到物体</p>';
                }
                
                $('#detectionStats').html(html);
                
                // VLM Result Update
                if (data.vlm_result) {
                    const res = data.vlm_result;
                    // Check if it's a new result by comparing timestamp or just update
                    // Ideally we should check if we already showed this one, but for now just show whatever is latest
                    
                    $('#vlmVerificationCard').show();
                    $('#vlmImage').attr('src', 'data:image/jpeg;base64,' + res.image_base64);
                    $('#vlmTime').text(res.timestamp);
                    $('#vlmClass').text('检测类别: ' + res.class_name);
                    
                    // Simple styling for YES/NO
                    let respText = res.vlm_response;
                    if (respText.includes('YES')) {
                        $('#vlmResponse').html('<span class="text-danger fw-bold">YES (违规确认)</span><br>' + respText);
                    } else if (respText.includes('NO')) {
                        $('#vlmResponse').html('<span class="text-success fw-bold">NO (误报排除)</span><br>' + respText);
                    } else {
                        $('#vlmResponse').text(respText);
                    }
                }

                // 摄像头警告处理：检测到指定类别时在顶部状态栏显示（只展示新增），且仅当 alertsEnabled 为 true
                if (data.warnings && data.warnings.length > 0) {
                    if (!window._cameraWarningsCount) window._cameraWarningsCount = 0;
                    if (data.warnings.length > window._cameraWarningsCount) {
                        const newList = data.warnings.slice(window._cameraWarningsCount).slice(-5).reverse();
                        const messages = newList.map(w => `帧 ${w.frame}: ${w.name} (${w.timestamp})`);
                        showAlerts(messages);
                        window._cameraWarningsCount = data.warnings.length;
                        // 将实时统计卡设为警示样式
                        $('.stats-card').addClass('has-alert');
                    }
                } else {
                    // 无警报时清除顶部与样式
                    if (window._cameraWarningsCount && window._cameraWarningsCount > 0) {
                        window._cameraWarningsCount = 0;
                    }
                    clearAlerts();
                    $('.stats-card').removeClass('has-alert');
                }
            }
        });
    }
    
    // 截图
    function captureFrame() {
        $.ajax({
            url: '/capture_frame',
            method: 'GET',
            success: function(data) {
                if (data.success) {
                    // 显示截图
                    $('#modalImage').attr('src', data.image_data);
                    $('#resultModal').modal('show');
                    
                    updateStatus('截图已保存: ' + data.filename, 'success');
                } else {
                    updateStatus('截图失败: ' + data.message, 'danger');
                }
            },
            error: function() {
                updateStatus('截图请求失败', 'danger');
            }
        });
    }
    
    // 上传图像
    function uploadImage() {
        const formData = new FormData();
        const imageFile = $('#imageFile')[0].files[0];
        
        if (!imageFile) {
            updateStatus('请选择图像文件', 'warning');
            return;
        }
        
        formData.append('file', imageFile);
        
        // 显示加载状态
        const originalText = $('#imageUploadForm button').html();
        $('#imageUploadForm button').html('<span class="loading"></span> 处理中...');
        $('#imageUploadForm button').prop('disabled', true);
        
        $.ajax({
            url: '/upload_image',
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(data) {
                if (data.success) {
                    // 显示结果
                    $('#processedImage').attr('src', data.image_data);
                    $('#imageStats').html(data.stats);
                    $('#downloadImage').attr('href', '/download/' + data.filename);
                    $('#imageResult').show();
                    updateStatus('图像处理完成', 'success');

                    // 如果存在警告，顶部显示警报（受 alertsEnabled 控制）
                    if (data.warnings && data.warnings.length > 0) {
                        const messages = data.warnings.slice(-5).reverse().map(w => `${w.name} (${w.timestamp})`);
                        showAlerts(messages);
                        $('.stats-card').addClass('has-alert');
                    }
                } else {
                    updateStatus('图像处理失败: ' + data.error, 'danger');
                }
            },
            error: function(xhr) {
                updateStatus('上传失败: ' + xhr.responseJSON?.error || '未知错误', 'danger');
            },
            complete: function() {
                // 恢复按钮状态
                $('#imageUploadForm button').html(originalText);
                $('#imageUploadForm button').prop('disabled', false);
            }
        });
    }
    
    // 上传视频
    function uploadVideo() {
        const formData = new FormData();
        const videoFile = $('#videoFile')[0].files[0];
        
        if (!videoFile) {
            updateStatus('请选择视频文件', 'warning');
            return;
        }
        
        formData.append('file', videoFile);
        
        // 显示加载状态
        const originalText = $('#videoUploadForm button').html();
        $('#videoUploadForm button').html('<span class="loading"></span> 处理中...');
        $('#videoUploadForm button').prop('disabled', true);
        
        $.ajax({
            url: '/upload_video',
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(data) {
                if (data.success) {
                    const filename = data.filename;
                    const streamUrl = data.stream_url;

                    // 显示流式视频
                    $('#processedVideoFrame').attr('src', streamUrl + '?' + new Date().getTime());
                    $('#videoResult').show();
                    updateStatus('开始流式播放上传视频', 'success');

                    // 设置下载日志链接（日志名: detections_<filename>.json）
                    $('#downloadVideoFrame').attr('href', '/download/detections_' + filename + '.json');

                    // 开始轮询统计
                    startVideoStatsPolling(filename);
                } else {
                    updateStatus('视频处理失败: ' + data.error, 'danger');
                }
            },
            error: function(xhr) {
                updateStatus('上传失败: ' + xhr.responseJSON?.error || '未知错误', 'danger');
            },
            complete: function() {
                // 恢复按钮状态
                $('#videoUploadForm button').html(originalText);
                $('#videoUploadForm button').prop('disabled', false);
            }
        });
    }

    // 上传视频后的轮询与警告处理
    let videoStatsInterval = null;
    let currentVideoFile = null;

    let lastWarningsCount = 0;

    function startVideoStatsPolling(filename) {
        stopVideoStatsPolling();
        currentVideoFile = filename;
        lastWarningsCount = 0;
        videoStatsInterval = setInterval(() => fetchVideoStats(filename), 200);
    }

    function stopVideoStatsPolling() {
        if (videoStatsInterval) {
            clearInterval(videoStatsInterval);
            videoStatsInterval = null;
        }
        if (currentVideoFile) {
            // 请求服务器清理会话
            $.ajax({ url: '/stop_video_session/' + currentVideoFile, method: 'POST' });
            currentVideoFile = null;
        }
        lastWarningsCount = 0;
        $('#downloadVideoFrame').hide();
    }

    function fetchVideoStats(filename) {
        $.ajax({
            url: '/video_stats/' + filename,
            method: 'GET',
            success: function(data) {
                let html = `<p><strong>FPS:</strong> ${data.fps || 0}</p><p><strong>总检测数:</strong> ${data.total || 0}</p>`;
                if (data.by_class && Object.keys(data.by_class).length > 0) {
                    html += '<ul>';
                    for (const [className, count] of Object.entries(data.by_class)) {
                        html += `<li>${className}: ${count}</li>`;
                    }
                    html += '</ul>';
                } else {
                    html += '<p class="text-muted">未检测到物体</p>';
                }
                $('#videoStats').html(html);

                // 处理警告（仅在 alertsEnabled 并且有新增警告时弹窗），避免重复弹出
                if (data.warnings && data.warnings.length > 0) {
                    if (data.warnings.length > lastWarningsCount) {
                        const newOnes = data.warnings.slice(lastWarningsCount).slice(-5).reverse();
                        const messages = newOnes.map(w => `帧 ${w.frame}: ${w.name} (${w.timestamp})`);
                        showAlerts(messages);
                        // 显示下载日志按钮
                        $('#downloadVideoFrame').show();
                        lastWarningsCount = data.warnings.length;
                        $('.stats-card').addClass('has-alert');
                    }
                } else {
                    lastWarningsCount = 0;
                    clearAlerts();
                    $('.stats-card').removeClass('has-alert');
                }
            }
        });
    }

    // 加载历史记录
    function loadHistory() {
        $.ajax({ url: '/history', method: 'GET', success: function(data) {
            if (data.success) {
                const tbody = $('#historyTable tbody');
                tbody.empty();
                data.rows.forEach(r => {
                    const tr = `<tr><td>${r.source}</td><td>${r.class_name}(${r.class_id})</td><td>${r.frame || ''}</td><td>${r.timestamp}</td></tr>`;
                    tbody.append(tr);
                });
            } else {
                console.error('history error', data.error);
            }
        }});
    }

    // 初始化历史记录和刷新按钮
    loadHistory();
    $('#refreshHistory').click(loadHistory);

    // 停止播放按钮
    $('#stopVideoStream').click(function() {
        // 停止轮询并清理会话
        stopVideoStatsPolling();
        $('#processedVideoFrame').attr('src', '');
        $('#videoResult').hide();
        updateStatus('已停止视频播放', 'info');
    });

    // 在顶部状态栏显示警报（不使用弹窗），alertsEnabled 控制是否显示
    function showAlerts(items) {
        if (!alertsEnabled) return;
        const alertsDiv = $('#statusAlerts');
        let html = '<ul class="alert-list mb-0">';
        items.forEach(it => { html += `<li>${it}</li>`; });
        html += '</ul>';
        alertsDiv.html(html);
        $('#statusAlert').removeClass('alert-info').addClass('alert-danger');
    }

    function clearAlerts() {
        $('#statusAlerts').empty();
        $('#statusAlert').removeClass('alert-danger').addClass('alert-info');
    }
    
    // 更新状态信息
    function updateStatus(message, type = 'info') {
        const icons = {
            'info': 'info-circle',
            'success': 'check-circle',
            'warning': 'exclamation-triangle',
            'danger': 'times-circle'
        };
        
        const alertClass = `alert-${type}`;
        const icon = icons[type] || 'info-circle';
        
        // 更改状态栏的视觉样式，但保持内部结构不被覆盖
        $('#statusAlert').removeClass('alert-info alert-success alert-warning alert-danger').addClass(alertClass);
        $('#statusMessage').html(`<i class="fas fa-${icon}"></i> ${message}`);

        // 非 info 类型短暂显示后恢复默认信息（不清除报警列表）
        if (type !== 'info') {
            setTimeout(() => {
                $('#statusAlert').removeClass('alert-success alert-warning alert-danger').addClass('alert-info');
                $('#statusMessage').text('系统就绪 | 模型: lab_safety_detection6');
            }, 3000);
        }
    }
    
    // 页面离开时停止摄像头
    $(window).on('beforeunload', function() {
        if (cameraActive) {
            $.ajax({
                url: '/stop_camera',
                method: 'GET',
                async: false // 同步请求，确保执行完成
            });
        }
        if (typeof stopVideoStatsPolling === 'function') {
            stopVideoStatsPolling();
        }
    });

    // --- 场景分析与安全要求 ---
    
    // 获取当前安全要求
    function loadRequirements() {
        $.ajax({
            url: '/get_requirements',
            method: 'GET',
            success: function(data) {
                if (data.vector && data.vector.length === 7) {
                    updateRequirementsUI(data.vector);
                }
            }
        });
    }
    loadRequirements();

    function updateRequirementsUI(vector) {
        $('.req-check').each(function() {
            const idx = $(this).data('idx');
            if (idx < vector.length) {
                 $(this).prop('checked', vector[idx] === 1);
            }
        });
        $('#vectorDisplay').text(JSON.stringify(vector));
    }

    // 场景分析表单提交
    $('#analysisForm').submit(function(e) {
        e.preventDefault();
        const formData = new FormData();
        const file = $('#analysisFile')[0].files[0];
        const desc = $('#analysisDesc').val();

        if (!file) {
            updateStatus('请选择图片', 'warning');
            return;
        }

        formData.append('file', file);
        formData.append('description', desc);

        const btn = $(this).find('button');
        const originalText = btn.html();
        btn.html('<span class="spinner-border spinner-border-sm"></span> 正在分析...').prop('disabled', true);

        $.ajax({
            url: '/analyze_scene',
            method: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(data) {
                if (data.success) {
                    updateRequirementsUI(data.vector);
                    $('#analysisText').text(data.raw_response); // 显示 VLM 返回的文本
                    $('#analysisResult').show();
                    updateStatus('场景分析完成，安全要求已更新', 'success');
                } else {
                    updateStatus('分析失败: ' + (data.error || '未知错误'), 'danger');
                }
            },
            error: function(xhr) {
                updateStatus('请求失败', 'danger');
            },
            complete: function() {
                btn.html(originalText).prop('disabled', false);
            }
        });
    });

    // 手动更新要求
    $('#updateRequirements').click(function() {
        const vector = [0, 0, 0, 0, 0, 0, 0];
        $('.req-check').each(function() {
            const idx = $(this).data('idx');
            if (idx < 7) {
                 vector[idx] = $(this).is(':checked') ? 1 : 0;
            }
        });

        $.ajax({
            url: '/update_requirements',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ vector: vector }),
            success: function(data) {
                if (data.success) {
                    $('#vectorDisplay').text(JSON.stringify(data.vector));
                    updateStatus('安全要求已手动更新', 'success');
                }
            },
            error: function() {
                updateStatus('更新失败', 'danger');
            }
        });
    });
});
