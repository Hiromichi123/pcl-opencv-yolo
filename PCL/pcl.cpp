// 1.读写pcd文件
// 1.1 读取点云数据
pcl::PCDReader reader;
pcd_reader.read("xxx.pcd", *cloud);
// 或者
pcl::io::loadPCDFile("xxx.pcd", *cloud);

// 1.2 保存点云数据
pcl::PCDWriter writer;
writer.write<pcl::PointT>("xxx.pcd", *cloud);
// 或者
pcl::io::savePCDFileASCII("xxx.pcd", *cloud);

// 2.滤波、降采样
// 2.1 体素滤波pcl::VoxelGrid
// 将空间划分为一定体积的网格，每个网格内的点用体素中心的点代替
pcl::VoxelGrid<pcl::PointT> vg;
vg.setInputCloud(cloud);
vg.setLeafSize(0.01f, 0.01f, 0.01f); // 网格大小
vg.filter(*cloud_filtered);

// 2.2 直通滤波pcl::PassThrough
// 保留点云中某一范围内的点，xyz方向都可设置
pcl::PassThrough<pcl::PointT> pass;
pass.setInputCloud(cloud);
pass.setFilterFieldName("z");
pass.setFilterLimits(0.0, 1.0);
pass.filter(*cloud_filtered);

// 2.3 半径滤波pcl::RadiusOutlierRemoval
// 去除点云中的离群点
pcl::RadiusOutlierRemoval<pcl::PointT> outrem;
outrem.setInputCloud(cloud);
outrem.setRadiusSearch(0.8);
outrem.setMinNeighborsInRadius(2);
outrem.filter(*cloud_filtered);

// 2.4 离群点滤波pcl::StatisticalOutlierRemoval
// 去除点云中的离群点
pcl::StatisticalOutlierRemoval<pcl::PointT> sor;
sor.setInputCloud(cloud);
sor.setMeanK(50);
sor.setStddevMulThresh(1.0);
sor.filter(*cloud_filtered);

// 3.聚类与分割
// 3.1 RANSAC（Random Sample Consensus）随机抽样一致性算法，最小二乘法升级版，用于拟合模型，如平面、直线等
// 在数据集中随机抽取最少可以拟合出模型的数据集，然后计算模型与数据集的拟合度，重复迭代直到找到最佳模型
// 3.1.1 平面分割
pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
pcl::SACSegmentation<pcl::PointT> seg;
seg.setOptimizeCoefficients(true);
seg.setModelType(pcl::SACMODEL_PLANE); // 设置平面模型
seg.setMethodType(pcl::SAC_RANSAC); // 设置RANSAC算法
seg.setMaxIterations(1000); // 最大迭代次数
seg.setDistanceThreshold(0.01); // 容差0.01m
seg.setInputCloud(cloud);
seg.segment(*inliers, *coefficients);

// 3.1.2 法线估计
// 估计点云中每个点的法线
pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>); // 法线点云
pcl::NormalEstimation<pcl::PointT, pcl::Normal> ne;
ne.setInputCloud(cloud);
pcl::search::KdTree<pcl::PointT>::Ptr tree(new pcl::search::KdTree<pcl::PointT>());
ne.setSearchMethod(tree); // 设置搜索方法
ne.setKSearch(0.03); // 设置搜索半径0.03m
ne.compute(*cloud_normals);

// 3.1.3 索引提取
// 从点云中提取指定索引的点
pcl::PointIndices::Ptr inliers(new pcl::PointIndices()); // 索引
pcl::ExtractIndices<pcl::PointXYZ> extract;
extract.setInputCloud(cloud);
extract.setIndices(inliers);
extract.setNegative(false); // true提取指定索引外的点，false提取指定索引的点
extract.filter(*cloud_filtered);

// 4.可视化
// 法一：会造成程序暂停。需要先在点云窗口按w键，再按q键退出
pcl::visualization::CloudViewer viewer("Cloud Viewer");
viewer.showCloud(cloud);
while (!viewer.wasStopped()) {}

// 法二：不会造成程序中断
pcl::visualization::cloudViewer viewer("Cloud Viewer");
...
while(!viewer.wasStopped()) {
    ...
    viewer.showCloud(cloud);
    ...
}
...
