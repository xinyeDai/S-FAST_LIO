#ifndef USE_IKFOM_H1
#define USE_IKFOM_H1

#include <vector>
#include <cstdlib>
#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "common_lib.h"
#include "sophus/so3.hpp"

//该hpp主要包含：状态变量x，输入量u的定义，以及正向传播中相关矩阵的函数

//24维的状态量x  （除了原来的18维，fastlio2中增加了旋转的外参和平移的外参6个维度，也就是雷达和IMU之间的旋转平移）状态量X的定义的顺序和论文有差异，为了方便后续计算残差的雅可比矩阵，这个顺序比较便利
struct state_ikfom
{
	Eigen::Vector3d pos = Eigen::Vector3d(0,0,0);  							// 位置
	Sophus::SO3<double> rot = Sophus::SO3<double>(Eigen::Matrix3d::Identity());			// 旋转
	Sophus::SO3<double> offset_R_L_I = Sophus::SO3<double>(Eigen::Matrix3d::Identity());		// 雷达和IMU之间的旋转外参
	Eigen::Vector3d offset_T_L_I = Eigen::Vector3d(0,0,0);						// 雷达和IMU之间的平移外参
	Eigen::Vector3d vel = Eigen::Vector3d(0,0,0);							// 速度
	Eigen::Vector3d bg = Eigen::Vector3d(0,0,0);							// 角速度偏置
	Eigen::Vector3d ba = Eigen::Vector3d(0,0,0);							// 加速度偏执
	Eigen::Vector3d grav = Eigen::Vector3d(0,0,-G_m_s2);						// 重力向量
};


//输入u  6维的输入（加速度和角速度）
struct input_ikfom
{
	Eigen::Vector3d acc = Eigen::Vector3d(0,0,0);
	Eigen::Vector3d gyro = Eigen::Vector3d(0,0,0);
};


//噪声协方差Q的初始化(对应公式(8)的Q, 在IMU_Processing.hpp中使用) 每三个维度等于一个系数乘单位阵
Eigen::Matrix<double, 12, 12> process_noise_cov()
{
	Eigen::Matrix<double, 12, 12> Q = Eigen::MatrixXd::Zero(12, 12);
	Q.block<3, 3>(0, 0) = 0.0001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(3, 3) = 0.0001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(6, 6) = 0.00001 * Eigen::Matrix3d::Identity();
	Q.block<3, 3>(9, 9) = 0.00001 * Eigen::Matrix3d::Identity();

	return Q;
}

//对应公式(2) 中的f  输入一个状态量的结构体s, 以及测量值的加速度和角速度in
Eigen::Matrix<double, 24, 1> get_f(state_ikfom s, input_ikfom in)	
{
// 对应顺序为速度(3)，角速度(3),外参T(3),外参旋转R(3)，加速度(3),角速度偏置(3),加速度偏置(3),位置(3)，与论文公式顺序不一致
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();  // 初始化为0
	Eigen::Vector3d omega = in.gyro - s.bg;		// 输入的imu的角速度(也就是实际测量值) - 估计的bias值(对应公式的第1行)
	Eigen::Vector3d a_inertial = s.rot.matrix() * (in.acc - s.ba);		//  输入的imu的加速度，先转到世界坐标系（对应公式的第3行）

	for (int i = 0; i < 3; i++)
	{
		res(i) = s.vel[i];		//速度（对应公式第2行） 0 1 2行
		res(i + 3) = omega[i];		//角速度（对应公式第1行） 3 4 5行 
		res(i + 12) = a_inertial[i] + s.grav[i];		//加速度（对应公式第3行）
	}

	return res;
}

//对应公式(7)的Fx  注意该矩阵没乘dt，没加单位阵
Eigen::Matrix<double, 24, 24> df_dx(state_ikfom s, input_ikfom in)
{
	Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
	cov.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();	//对应公式(7)第2行第3列   I
	Eigen::Vector3d acc_ = in.acc - s.ba;   	//测量加速度 = a_m - bias	

	cov.block<3, 3>(12, 3) = -s.rot.matrix() * Sophus::SO3<double>::hat(acc_);		//对应公式(7)第3行第1列
	cov.block<3, 3>(12, 18) = -s.rot.matrix(); 				//对应公式(7)第3行第5列 

	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity();		//对应公式(7)第3行第6列   I
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();		//对应公式(7)第1行第4列 (简化为-I，因为A阵和Δt比较小) 
	return cov;
}

//对应公式(7)的Fw  注意该矩阵没乘dt
Eigen::Matrix<double, 24, 12> df_dw(state_ikfom s, input_ikfom in)
{
	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
	cov.block<3, 3>(12, 3) = -s.rot.matrix();					//对应公式(7)第3行第2列  -R 
	cov.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();		//对应公式(7)第1行第1列  -A(w dt)简化为-I
	cov.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();		//对应公式(7)第4行第3列  I
	cov.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();		//对应公式(7)第5行第4列  I
	return cov;
}

#endif
