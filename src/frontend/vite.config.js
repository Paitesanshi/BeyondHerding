import {
	defineConfig
} from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
	plugins: [vue()],
	server: {
		proxy: {
			// 匹配以 /api 开头的请求路径
			'/api': {
				// 目标服务器地址
				target: 'http://127.0.0.1:8000/',
				// 允许跨域
				changeOrigin: true,
				// 重写路径，去除 /api 前缀
				rewrite: (path) => path.replace(/^\/api/, ''),
			},
		},
	},
})