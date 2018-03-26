# Easy-Image-Stitching
Assignment 03 for Computer Vision in ZJU 2017 Fall

采用 Python 2.7 + OpenCV 2.4 + NumPy 。

 <script type="text/x-mathjax-config">
      MathJax.Hub.Config({
      extensions: ["tex2jax.js"],
      jax: ["input/TeX", "output/HTML-CSS"],
      tex2jax: {
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
        displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
        processEscapes: true
      },
      "HTML-CSS": { availableFonts: ["TeX"] }
      });
</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML'></script>

# 基本算法与流程
输入: n 个有序图像，图像编号从 1 开始，到 n 结束
1. 取中间的参考图像为参考图像 ref ，该图像不做变形。该图像在 1~n 的编号中为 (n/2)+1 。
2. 以 ref 为分界线，在两侧，分别以两张相邻图片为一组，使离 ref 较远的图片变形去符合离 ref 较近的图片，计算 Homography 矩阵。记 $H_{(A, B)}$ 为图片 A 为了符合图片 B 进行变形的 Homography 矩阵。则共得到 n-1 个 Homography 矩阵：

$$
H_{(1,2)}, H_{(2,3)}, \cdots, H_{(ref-1, ref}, H_{(ref+1, ref)}, H_{(ref+2, ref+1)}, \cdots ,H_{(n, n-1)}
$$

    1. 记离 ref 较远的图片为 imgA ，离 ref 较近的图片为 imgB ， imgA 为了符合 imgB 变形。
    2. 分别计算两图像的 SIFT 特征，得到关键点集和特征集。
    3. 使用 kNN 算法在两个特征向量集间进行匹配，为了去除 false-positive 的匹配，可引入 Lowe's ratio ，最后在匹配点集中使用 RANSAC 算法计算 Homography 矩阵。
    
3. 对第 2 步得到的矩阵进行处理。因为

$$
H_{(1, ref)}=H_{(1,2)} \times H_{(2,3)} \times \cdots \times H_{(ref-1, ref)}
$$

所以对第 2 步中的矩阵进行连乘。得到使图片变换到 ref 的坐标系的 n-1 个 Homography 矩阵：

$$
H_{(1, ref)}, H_{(2, ref)}, \cdots, H_{(ref-1, ref)}, H_{(ref+1, ref)}, \cdots, H_{(n-1, ref)}
$$

4. 将每幅图像 i 都乘以它对应的 Homography 矩阵 $H_{(i, ref)}$ 并且将这些图片拼接起来，就得到了最终的拼接图像。

输出：拼接图像
