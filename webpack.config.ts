import HtmlWebpackPlugin from 'html-webpack-plugin';
import path from 'path';

const port = process.env.PORT || 3000;

var config: any = {
    entry: './src/index.tsx',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'bundle.[fullhash].js'
    },
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                exclude: /node_modules/,
                use: ['ts-loader'],
            },
            {
                test: /\.(js)$/,
                exclude: /node_modules/,
                use: ['babel-loader']
            },
            {
                test: /\.css$/,
                use: ['style-loader', 'css-loader'],
            }
        ]
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js']
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: 'public/index.html',
            favicon: 'public/favicon.ico'
        })
    ]
};

module.exports = (env: any, argv: any) => {

    if(argv.mode == 'development') {
        config.devtool = 'eval-source-map';
        config.devServer = {
            static: path.resolve(__dirname, 'dist'),
            port: port,
            compress: true,
            open: true
        };
    }
    if(argv.mode == 'production') {
        config.devtool = false;
        config.performance = {
            hints: false,
            maxEntrypointSize: 256000,
            maxAssetSize: 256000
        };
    }

    return config;
};