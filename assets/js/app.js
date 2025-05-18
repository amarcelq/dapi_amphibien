import $ from 'jquery'

$(_ => {
    console.log("Jquery loaded");
    // hide every element using jquery instead of class:
    $(".hidden").hide().removeClass("hidden")

    // enable drag and drop area
    $('#drop-input').on('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).addClass('dragover');
    });

    $('#drop-input').on('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).removeClass('dragover');
    });

    $('#drop-input').on('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).removeClass('dragover');

        var files = e.originalEvent.dataTransfer.files;
        if (files.length === 0) return;
        var file = files[0];
        // TODO: display error
        if (file.type !== 'audio/wav' && !(file.name.endsWith('.wav') || file.name.endsWith('.WAV'))) {
            console.log("Not supported files");
            return
        }

        var formData = new FormData();
        formData.append('csrfmiddlewaretoken', $('input[name=csrfmiddlewaretoken]').val());
        formData.append('file', file);

        $.ajax({
            url: '/wav',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#drop-input, .title, .subtitle').hide(100);
            }
        });

        
    });
})