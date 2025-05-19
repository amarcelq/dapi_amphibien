import $ from 'jquery'
import { show_big_waveform } from './audio'

const fadeTime = 150

function updateProgressBar (percentage, text) {
  const gradient = `conic-gradient(var(--color-green) ${percentage}%, rgba(47, 158, 68, 0) ${
    percentage + 0.01
  }%)`
  $('#yellow .progress-bar').css('background', gradient)
  $('#yellow .process-info').text(text)
}

// For dev purposes have random progress to manually advance
var progress = 0
function dummy_progress () {
  progress += Math.floor(Math.random() * 50) + 1
  const progress_texts = [
    'Loading...',
    'Processing...',
    'Almost there...',
    'Working on it...',
    'Hang tight...'
  ]
  const progress_text =
    progress_texts[Math.floor(Math.random() * progress_texts.length)]
  updateProgressBar(progress, progress_text)

  // if progress is >=100 do the steps the progres route would do
  if (progress >= 100) {
    // hide progress stuff
    $('#yellow .process, #yellow .progress-bar, #yellow .process-info').hide()
    // show audio stuff (at this point dummy)
    $('#yellow .file-bar, #yellow .text, #yellow .tiles').fadeIn(fadeTime)
    show_big_waveform(
      '/media/sessions/p8j9rvbb3b135e4cgr2hfd46bnufs7uq/upload.wav'
    )
  }
}

function dummy_file () {
  $('#yellow #drop-input, #yellow .title, #yellow .subtitle').fadeOut(fadeTime)
  $('#yellow .process, #yellow .progress-bar, #yellow .process-info').fadeIn(
    fadeTime
  )
}

window.dummy_progress = dummy_progress
window.dummy_file = dummy_file

$(_ => {
  console.log('Jquery loaded')
  // hide every element using jquery instead of class:
  $('.hidden').hide().removeClass('hidden')

  // enable drag and drop area
  $('#drop-input').on('dragover', function (e) {
    e.preventDefault()
    e.stopPropagation()
    $(this).addClass('dragover')
  })

  $('#drop-input').on('dragleave', function (e) {
    e.preventDefault()
    e.stopPropagation()
    $(this).removeClass('dragover')
  })

  $('#drop-input').on('drop', function (e) {
    e.preventDefault()
    e.stopPropagation()
    $(this).removeClass('dragover')

    var files = e.originalEvent.dataTransfer.files
    if (files.length === 0) return
    var file = files[0]
    // TODO: display error
    if (
      file.type !== 'audio/wav' &&
      !(file.name.endsWith('.wav') || file.name.endsWith('.WAV'))
    ) {
      console.log('Not supported files')
      return
    }

    var formData = new FormData()
    formData.append(
      'csrfmiddlewaretoken',
      $('input[name=csrfmiddlewaretoken]').val()
    )
    formData.append('file', file)

    $.ajax({
      url: '/wav',
      type: 'POST',
      data: formData,
      processData: false,
      contentType: false,
      success: function (response) {
        $('#yellow #drop-input, #yellow .title, #yellow .subtitle').fadeOut(
          fadeTime
        )
        $(
          '#yellow .process, #yellow .progress-bar, #yellow .process-info'
        ).fadeIn(fadeTime)
      }
    })
  })
  updateProgressBar(0)

  $('.tile .main .more button').on('click', e => {
    const $tile = $(e.currentTarget).closest('.tile')
    if ($tile.hasClass('open')) {
      $tile.removeClass('open')
    } else {
      $('.tile').removeClass('open')
      $tile.addClass('open')
    }
  })

  $('.tile .main .top button.sound').on('click', e => {
    let slider = $(e.currentTarget).closest('.top').find('.slider')
    if (slider.is(':visible')) slider.hide(100)
    else slider.show(100)
  })

  $(document).on('click', function (e) {
    slide = $(e.target).closest('.slider, button.sound')
    if (slide.length == 0) {
      $('.tile .main .top .slider').hide(100)
    }
  })
})
