import * as esbuild from 'esbuild'
import path from 'path'

let minify = false
let sourcemap = true
let watch = true

if (process.env.NODE_ENV === 'production') {
  minify = true
  sourcemap = false
  watch = false
}

const config = {
  entryPoints: ['./css/app.css'],
  outfile: path.resolve('../public/css/app.css'),
  bundle: true,
  loader: { '.css': 'css'},
  minify: minify,
  sourcemap: sourcemap,
}

if (watch) {
  let context = await esbuild.context({ ...config, logLevel: 'info' })
  await context.watch()
} else {
  esbuild.build(config)
}
