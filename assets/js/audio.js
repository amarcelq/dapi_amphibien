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
        normalize: true
        // the Minimap takes all the same options as the WaveSurfer itself
      })
    ]
  })

  ws.on('interaction', () => {
    ws.play()
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
}
