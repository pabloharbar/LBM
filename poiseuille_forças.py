import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Parâmetros do domínio
# ===============================
H = 32
Ny = 2 * H + 1
Nx = H
Nt_max = 25000
tempos_saida = [Nt_max//4, Nt_max//2, 3*Nt_max//4, Nt_max]

# ===============================
# Parâmetros LBM conhecidos: Reynolds e tau
# ===============================
Re = 100.0
tau = 0.8
rho0 = 1.0

# viscosidade lattice
nu = (tau - 0.5) / 3
u_max = Re * nu / (2*H)    # velocidade central Poiseuille
omega = 1.0 / tau

print(f"nu={nu:.6f}, u_max={u_max:.6f}, tau={tau:.2f}")

# ===============================
# Constantes D2Q9
# ===============================
w = np.array([4/9] + [1/9]*4 + [1/36]*4)
c = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1],
              [1,1], [-1,1], [-1,-1], [1,-1]]).astype(int)
opposite = [0,3,4,1,2,7,8,5,6]

# ===============================
# Função de equilíbrio
# ===============================
def feq(rho, ux, uy):
    feq_local = np.zeros((9, Ny, Nx))
    u2 = ux**2 + uy**2
    for i in range(9):
        cu = 3*(c[i,0]*ux + c[i,1]*uy)
        feq_local[i] = w[i]*rho*(1 + cu + 0.5*cu**2 - 1.5*u2)
    return feq_local

# ===============================
# Inicialização
# ===============================
rho = np.ones((Ny, Nx))
ux = np.zeros((Ny, Nx))
uy = np.zeros((Ny, Nx))
f = feq(rho, ux, uy)

# paredes
is_solid = np.zeros((Ny, Nx), dtype=bool)
is_solid[0, :] = True
is_solid[-1, :] = True

# ===============================
# Força equivalente a deltaP
# ===============================
Fx = 2 * nu * u_max / H**2   # força lattice
Fy = 0.0

def force_term(rho, ux, uy, Fx, Fy):
    Fx_mat = Fx * np.ones_like(ux)
    Fy_mat = Fy * np.ones_like(uy)
    S = np.zeros((9, Ny, Nx))
    for i in range(9):
        ci_dot_u = c[i,0]*ux + c[i,1]*uy
        ci_dot_F = c[i,0]*Fx_mat + c[i,1]*Fy_mat
        S[i] = w[i]*(3*ci_dot_F + 9*ci_dot_u*ci_dot_F - 3*(ux*Fx_mat + uy*Fy_mat))
    return S

# ===============================
# Perfil teórico de Poiseuille
# ===============================
y = np.arange(Ny)
y_central = y - H
u_teorico = (Fx * (H**2 - y_central**2)) / (2 * nu)

# ===============================
# Loop principal com critério de parada (MSE)
# ===============================
x_central = Nx//2
mse_history = []
perfis = {}

for t in range(1, Nt_max+1):
    # --- Macroscópicos ---
    rho = f.sum(axis=0)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    for i in range(9):
        ux += f[i] * c[i,0]
        uy += f[i] * c[i,1]
    ux /= rho
    uy /= rho

    # --- Colisão com força ---
    feq_local = feq(rho, ux, uy)
    S = force_term(rho, ux, uy, Fx, Fy)
    f = f - omega*(f - feq_local) + S

    # --- Streaming ---
    for i in range(9):
        f[i] = np.roll(np.roll(f[i], c[i,1], axis=0), c[i,0], axis=1)

    # --- Bounce-back nas paredes ---
    for i in range(9):
        f[i][is_solid] = f[opposite[i]][is_solid]

    # --- Periodicidade em x ---
    for i in range(9):
        f[i][:,0] = f[i][:,-2]
        f[i][:,-1] = f[i][:,1]

    # --- Cálculo do MSE (sem paredes) ---
    mask = ~is_solid[:, x_central]
    mse = np.mean((ux[mask, x_central] - u_teorico[mask])**2)
    mse_history.append(mse)

    # --- Imprimir cada 1000 passos ---
    if t % 1000 == 0:
        print(f"Iteração {t}: MSE = {mse:.6e}")

    # --- Critério de convergência ---
    if mse < 1e-6:
        print(f"Convergência alcançada em t = {t}, MSE = {mse:.6e}")
        perfis[t] = ux[:, x_central].copy()
        break

    # --- Salvar perfil temporal apenas em x_central ---
    if t in tempos_saida:
        perfis[t] = ux[:, x_central].copy()

print("Simulação finalizada.")

# ===============================
# Plot perfil no centro (x_central)
# ===============================
plt.figure(figsize=(6,5))
for t in perfis:
    plt.plot(perfis[t], y, label=f"t={t}")
plt.plot(u_teorico, y, 'k--', label="Teórico")
plt.xlabel("u_x")
plt.ylabel("y")
plt.title(f"Perfil em x = {x_central}")
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# Curva de convergência (MSE)
# ===============================
plt.figure(figsize=(6,4))
plt.semilogy(mse_history)
plt.xlabel("Iteração")
plt.ylabel("MSE")
plt.title("Evolução do erro quadrático médio (MSE)")
plt.grid(True)
plt.show()

# ===============================
# Heatmap da magnitude da velocidade
# ===============================
vel_mag = np.sqrt(ux**2 + uy**2)
plt.figure(figsize=(12,4))
plt.imshow(vel_mag, origin='lower', cmap='viridis', extent=[0, Nx-1, 0, Ny-1], aspect='auto')
plt.colorbar(label='|u|')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Magnitude da velocidade |u| em t = {t}')
plt.show()
