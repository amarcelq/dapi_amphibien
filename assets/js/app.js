import $ from 'jquery'
import {
  show_big_waveform,
  jump_to_position,
  show_tile_waveform
} from './audio'

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
  updateProgressBar((-1 / (0.23 * progress + 1) + 1) * 100, progress_text)

  // if progress is >=100 do the steps the progres route would do
  if (progress >= 100) {
    finish_process()
  }
}

function finish_process () {
  // hide progress stuff
  $('#yellow .process, #yellow .progress-bar, #yellow .process-info').hide()
  // show audio stuff (at this point dummy)
  $('#yellow .file-bar, #yellow .text, #yellow .tiles').fadeIn(fadeTime)
  // get results from end point
  $.getJSON('/result', function (data) {
    console.log('Result:', data)
    // Process the JSON result as needed
    show_big_waveform(data.main_audio.url)
    // preset tile
    const $tile = $(`body>.tile.PRESET`)
    const $snippet = $(`body>.sample.PRESET`)
    const $container = $('#yellow .tiles')
    for (const sample of data.samples) {
      // create new tile
      const $new = $tile.clone(true).removeClass('PRESET')
      $new.find('.main .top .name').text(sample.name)

      // first snipped is big one:
      const big_snippet = sample.snippets.shift()
      $new.data('start', big_snippet.start)
      $new.data('duration', big_snippet.duration)

      const $sample_container = $new.find('.side')
      // add snippets
      for (const snip of sample.snippets) {
        const $new_snip = $snippet.clone(true).removeClass('PRESET')
        $new_snip.find('.time span').text(ms_to_minute_string(snip.start))
        $new_snip.find('a.download').attr('href', snip.url)
        $new_snip.data('start', snip.start)
        $new_snip.data('duration', snip.duration)
        //   append to tile
        $new_snip.appendTo($sample_container)
      }
      //   at end append tile to DOM
      $new.appendTo($container)
      // show wave form(s)
      show_tile_waveform($new, big_snippet.url)
    }
  })
}

function dummy_file () {
  $('#yellow #drop-input, #yellow .title, #yellow .subtitle').fadeOut(fadeTime)
  $('#yellow .process, #yellow .progress-bar, #yellow .process-info').fadeIn(
    fadeTime
  )
}

window.dummy_progress = dummy_progress
window.dummy_file = dummy_file

function ms_to_minute_string (ms) {
  const minutes = Math.floor(ms / 60000)
  const seconds = Math.floor((ms % 60000) / 1000)
  return `${minutes}m ${seconds}s`
}

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
        pollProgress()
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
  // will be set on start, so only presets exist atm
  $('.sample.PRESET .time').on('click', e => {
    const $sample = $(e.currentTarget).closest('.sample')
    // set big waveform to specified position
    console.log($sample.data('start'), $sample.data('duration'))

    jump_to_position($sample.data('start'), $sample.data('duration'))
  })
})

function pollProgress () {
  const pollInterval = 500 // 1 second
  const $progress_name = $('#yellow .process')
  const $progress_desc = $('#yellow .process-info')
  let last_pogress = ''
  function fetchProgress () {
    $.getJSON('/progress', function (data) {
      console.log(data)
      $progress_name.text(data.name)
      $progress_desc.text(data.description)
      if (last_pogress != data.description) {
        last_pogress = data.description
        progress += 1
        updateProgressBar((-1 / (0.23 * progress + 1) + 1) * 100)
      }
      if (data.status === 'done') {
        updateProgressBar(1000000)
        finish_process()
      } else {
        setTimeout(fetchProgress, pollInterval)
      }
    })
  }

  fetchProgress()
}
