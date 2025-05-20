import $ from 'jquery'
// import { Howl, Howler } from 'howler';
import WaveSurfer from 'wavesurfer.js'
import Minimap from 'wavesurfer.js/dist/plugins/minimap.esm.js'

// see https://wavesurfer.xyz/examples/?minimap.js

export var ws = null

export function show_big_waveform (source_path) {
  var $wave = $('#yellow .file-bar .wave .form')
  var $map = $('#yellow .file-bar .wave .map')
  var $play_button = $('#yellow .file-bar > button')

  ws = WaveSurfer.create({
    container: $wave[0],
    waveColor: 'rgb(53, 53, 53)',
    progressColor: 'rgb(30, 30, 30)',
    url: source_path,
    minPxPerSec: 100,
    hideScrollbar: true,
    autoCenter: false,
    height: 'auto',
    normalize: true,
    plugins: [
      // Register the plugin
      Minimap.create({
        container: $map[0],
        height: '20',
        waveColor: '#999',
        progressColor: '#ddd',
        normalize: true,
        overlayColor: '#44444499'
        // the Minimap takes all the same options as the WaveSurfer itself
      })
    ]
  })

  ws.on('interaction', () => {
    ws.play()
    $play_button.removeClass('paused')
  })

  ws.on('pause', _ => {
    $play_button.addClass('paused')
  })

  ws.on('play', _ => {
    $play_button.removeClass('paused')
  })

  $play_button.on('click', e => {
    if ($play_button.hasClass('paused')) {
      $play_button.removeClass('paused')
      ws.play()
    } else {
      $play_button.addClass('paused')
      ws.pause()
    }
  })
  ws.on('audioprocess', () => {
    const currentTime = ws.getCurrentTime().toFixed(3)
    $('#yellow .file-bar .info .time span').text(currentTime + 's')
  })
}

export function show_tile_waveform (tile, source_path) {
  var $tile = $(tile)
  $tile.find('.main .wave *').remove()
  var wave = WaveSurfer.create({
    container: $tile.find('.main .wave')[0],
    waveColor: 'rgb(53, 53, 53)',
    progressColor: 'rgb(30, 30, 30)',
    url: source_path,
    minPxPerSec: 50,
    hideScrollbar: true,
    autoCenter: true,
    height: 'auto',
    normalize: true,
    dragToSeek: true,
    autoScroll: true
  })
  // volume handlers
  $tile.find('.main .top .slider input').on('input', e => {
    var slider = $(e.currentTarget)
    wave.setVolume(slider.val())
  })

  // play
  $tile.on('click', e => {
    wave.playPause()
  })
}

window.show_tile_waveform = show_tile_waveform

export function jump_to_position (position_in_ms, duration = null) {
  if (duration) {
    ws.play(position_in_ms / 1000, (position_in_ms + duration) / 1000)
  } else ws.play(position_in_ms / 1000)
}
