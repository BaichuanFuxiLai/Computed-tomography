import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from scipy import sparse
from scipy.sparse import linalg as splinalg
import os
import glob
from mpl_toolkits.mplot3d import Axes3D


# ----------------------------------------------------------------------
# (对应原理 - 射线追踪) 精确版 Siddon 算法：计算射线在每个体素中的路径长度
# ----------------------------------------------------------------------
def siddon_ray_path(source, receiver, grid_params):
    x_min, y_min, z_min = grid_params['x_min'], grid_params['y_min'], grid_params['z_min']
    dx, dy, dz = grid_params['dx'], grid_params['dy'], grid_params['dz']
    nx, ny, nz = grid_params['nx'], grid_params['ny'], grid_params['nz']

    src = np.array(source, dtype=np.float64)
    rec = np.array(receiver, dtype=np.float64)
    direction = rec - src

    if np.allclose(direction, 0):
        return []

    L = np.linalg.norm(direction)
    direction /= L

    # 计算进入和离开网格的参数 t
    t_min, t_max = -np.inf, np.inf
    for i, (s, d, gmin, gmax, dg) in enumerate(zip(src, direction, [x_min, y_min, z_min],
                                                   [x_min + nx * dx, y_min + ny * dy, z_min + nz * dz],
                                                   [dx, dy, dz])):
        if abs(d) < 1e-12:
            if s < gmin or s > gmax:
                return []
        else:
            t1 = (gmin - s) / d
            t2 = (gmax - s) / d
            tmin_i, tmax_i = min(t1, t2), max(t1, t2)
            t_min = max(t_min, tmin_i)
            t_max = min(t_max, tmax_i)
    if t_min >= t_max:
        return []

    t_min = max(t_min, 0.0)
    t_max = min(t_max, L)

    p_min = src + t_min * direction
    ix = int((p_min[0] - x_min) / dx)
    iy = int((p_min[1] - y_min) / dy)
    iz = int((p_min[2] - z_min) / dz)

    if direction[0] > 0:
        step_x, t_max_x, dt_x = 1, ((x_min + (ix + 1) * dx) - p_min[0]) / direction[0], dx / direction[0]
    elif direction[0] < 0:
        step_x, t_max_x, dt_x = -1, ((x_min + ix * dx) - p_min[0]) / direction[0], -dx / direction[0]
    else:
        step_x, t_max_x, dt_x = 0, np.inf, np.inf

    if direction[1] > 0:
        step_y, t_max_y, dt_y = 1, ((y_min + (iy + 1) * dy) - p_min[1]) / direction[1], dy / direction[1]
    elif direction[1] < 0:
        step_y, t_max_y, dt_y = -1, ((y_min + iy * dy) - p_min[1]) / direction[1], -dy / direction[1]
    else:
        step_y, t_max_y, dt_y = 0, np.inf, np.inf

    if direction[2] > 0:
        step_z, t_max_z, dt_z = 1, ((z_min + (iz + 1) * dz) - p_min[2]) / direction[2], dz / direction[2]
    elif direction[2] < 0:
        step_z, t_max_z, dt_z = -1, ((z_min + iz * dz) - p_min[2]) / direction[2], -dz / direction[2]
    else:
        step_z, t_max_z, dt_z = 0, np.inf, np.inf

    path_segments = []
    t = t_min
    while t < t_max - 1e-10:
        if t_max_x < t_max_y and t_max_x < t_max_z:
            t_next = t_max_x
            ix += step_x
            t_max_x += dt_x
        elif t_max_y < t_max_z:
            t_next = t_max_y
            iy += step_y
            t_max_y += dt_y
        else:
            t_next = t_max_z
            iz += step_z
            t_max_z += dt_z

        if t_next > t_max:
            t_next = t_max
        seg_length = (t_next - t) * np.linalg.norm(direction)
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            path_segments.append(((ix, iy, iz), seg_length))
        t = t_next

    return path_segments


# ----------------------------------------------------------------------
# (对应步骤 3 - 正演计算) 正演计算：理论走时和灵敏度矩阵
# ----------------------------------------------------------------------
def forward_calculation_with_siddon(slowness_model, events, stations, grid_params):
    nx, ny, nz = grid_params['nx'], grid_params['ny'], grid_params['nz']
    n_rays = len(events)
    n_voxels = nx * ny * nz

    theoretical_tt = np.zeros(n_rays)
    rows, cols, data = [], [], []

    print(f"正在进行 {n_rays} 条射线的正演计算...")
    for i in range(n_rays):
        src, rec = events[i], stations[i]
        segments = siddon_ray_path(src, rec, grid_params)
        ttime = 0.0
        for (ix, iy, iz), seg_len in segments:
            s = slowness_model[ix, iy, iz]
            ttime += seg_len * s
            rows.append(i)
            cols.append(ix * ny * nz + iy * nz + iz)
            data.append(seg_len)
        theoretical_tt[i] = ttime

    G = sparse.csr_matrix((data, (rows, cols)), shape=(n_rays, n_voxels))
    print("正演计算完成。")
    return theoretical_tt, G


# ----------------------------------------------------------------------
# (对应步骤 4 - 反演计算) 反演：求解模型修正量
# ----------------------------------------------------------------------
def inversion_with_regularization(G, residuals, lam=0.1):
    n_params = G.shape[1]
    I = sparse.identity(n_params, format='csr')
    G_reg = sparse.vstack([G, np.sqrt(lam) * I])
    d_reg = np.concatenate([residuals, np.zeros(n_params)])
    print(f"正在求解大型线性方程组 (G * m = d)...")
    # 修改：lsqr 返回一个包含解和残差范数的元组
    res = splinalg.lsqr(G_reg, d_reg, atol=1e-6, btol=1e-6)
    print("求解完成。")
    # 返回解和残差范数
    return res[0], res[3]


# ----------------------------------------------------------------------
# (对应步骤 1 - 数据准备与预处理) 数据准备与筛选函数
# ----------------------------------------------------------------------
def prepare_and_filter_data(file_path):
    print(f"正在读取和预处理文件: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8')
    data = df[['震源X/m', '震源Y/m', '震源Z/m', '台站X/m', '台站Y/m', '台站Z/m', '传播时间/ms']].copy()
    data.columns = ['sx', 'sy', 'sz', 'rx', 'ry', 'rz', 'tt']
    data.dropna(inplace=True)
    data['tt_sec'] = data['tt'] / 1000.0
    data = data[(data['tt_sec'] > 0.0) & (data['tt_sec'] < 1.0)]

    if data.empty:
        print("警告：经过筛选后，数据为空。无法进行层析成像。")
        return None, None, None

    events = data[['sx', 'sy', 'sz']].values
    stations = data[['rx', 'ry', 'rz']].values
    obs_tt = data['tt_sec'].values

    print(f"数据预处理完成，共筛选出 {len(obs_tt)} 条有效射线。")
    return events, stations, obs_tt


# ----------------------------------------------------------------------
# 最终的三维可视化：三维体渲染式热力图 (模拟效果)
# ----------------------------------------------------------------------
def visualize_3d_results(vel_final, grid_params, events, stations, file_root):
    nx, ny, nz = grid_params['nx'], grid_params['ny'], grid_params['nz']
    x_min, y_min, z_min = grid_params['x_min'], grid_params['y_min'], grid_params['z_min']

    grid_x = np.linspace(x_min, x_min + (nx - 1) * grid_params['dx'], nx)
    grid_y = np.linspace(y_min, y_min + (ny - 1) * grid_params['dy'], ny)
    grid_z = np.linspace(z_min, z_min + (nz - 1) * grid_params['dz'], nz)

    v_mean, v_std = vel_final.mean(), vel_final.std()
    # 调整归一化范围，以更好地突出异常
    norm = Normalize(vmin=v_mean - 2.0 * v_std, vmax=v_mean + 2.0 * v_std)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 沿着所有z_idx创建切片，模拟连续体渲染
    z_render_indices = np.arange(nz).astype(int)

    X, Y = np.meshgrid(grid_x, grid_y)

    # 绘制模拟体渲染的切片
    for z_idx in z_render_indices:
        # 使用 contourf 绘制每个Z平面上的速度场
        # cmap 和 norm 保持不变
        # alpha 进一步减小，因为切片更密集，需要更高透明度
        ax.contourf(X, Y, vel_final[:, :, z_idx].T, levels=50, zdir='z', offset=grid_z[z_idx], cmap=cm.jet, norm=norm,
                    alpha=0.08)  # 核心修改点：更密的切片，更小的alpha值

    # 移除震源和台站的散点图，如原始代码所示
    # ax.scatter(events[:,0], events[:,1], events[:,2], c='black', marker='o', label='Events', s=20)
    # ax.scatter(stations[:,0], stations[:,1], stations[:,2], c='green', marker='^', label='Stations', s=20)

    # 设置图表标签和标题
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"3D Volumetric Velocity Model for {file_root} (Simulated)")
    ax.set_zlim(grid_z.min(), grid_z.max())
    ax.view_init(elev=20, azim=-45)  # 可以调整视角，更好地观察三维效果

    # 添加颜色条
    # ScalarMappable 必须有 cmap 和 norm
    sm = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    sm.set_array([])  # 必须设置一个空的数组，否则会报错
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Velocity (m/s)")

    output_filename = f"{file_root}_3D_Volumetric_Result.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"--- 成功保存三维体渲染结果图像: {output_filename} ---")
    print(
        f"结果解释：您现在可以看到一个模拟的三维体渲染效果，不同深度、不同位置的速度异常通过颜色和透明度连续地展示出来。蓝色区域代表高速异常，红色区域代表低速异常。")


# ----------------------------------------------------------------------
# (主流程) 整个层析成像的循环迭代过程
# ----------------------------------------------------------------------
def process_file(file_path, iterations=5, lam=0.1):
    base_name = os.path.basename(file_path)
    file_root = os.path.splitext(base_name)[0]

    print(f"\n--- 开始处理文件: {file_path} ---")

    # 1. 数据准备
    events, stations, obs_tt = prepare_and_filter_data(file_path)
    if events is None:
        return

    # 2. 建立初始速度模型与网格参数化
    all_coords = np.vstack([events, stations])
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()

    nx, ny = 40, 40
    nz = 100

    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    dz = (z_max - z_min) / (nz - 1)

    grid_params = {'x_min': x_min, 'y_min': y_min, 'z_min': z_min, 'nx': nx, 'ny': ny, 'nz': nz, 'dx': dx, 'dy': dy,
                   'dz': dz}

    v0 = 3500.0
    slowness = np.full((nx, ny, nz), 1.0 / v0)

    print(f"初始模型：均匀速度 {v0} m/s")
    print(f"网格参数：nx={nx}, ny={ny}, nz={nz}")
    print(f"开始迭代反演，共 {iterations} 轮...")

    # 3. 反演迭代循环
    for it in range(iterations):
        print(f"\n--- 第 {it + 1}/{iterations} 轮 ---")

        # 每轮都重新进行正演计算，这是确保残差更新的关键
        theo_tt, G = forward_calculation_with_siddon(slowness, events, stations, grid_params)

        residuals = obs_tt - theo_tt
        print(f"本轮走时残差(ms)均方根: {np.sqrt(np.mean(residuals ** 2)) * 1000:.2f}")

        # 修改：inversion_with_regularization 现在返回残差范数
        delta_s, residual_norm = inversion_with_regularization(G, residuals, lam)

        slowness += delta_s.reshape((nx, ny, nz))
        slowness = np.clip(slowness, 1.0 / (v0 * 2.0), 1.0 / (v0 * 0.5))

    vel = 1.0 / slowness
    print("\n迭代完成！")

    vel_final = gaussian_filter(vel, sigma=1.0)

    # 调用最终的三维可视化函数
    visualize_3d_results(vel_final, grid_params, events, stations, file_root)


# ----------------------------------------------------------------------
# (主入口)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    files = glob.glob("*.csv")
    if not files:
        print("当前目录未找到 CSV 文件。请确保数据文件存在。")
    else:
        for f in files:
            process_file(f, iterations=5, lam=0.1)
        print("\n所有文件处理完成。")