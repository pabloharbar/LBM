import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Parâmetros do domínio
# ===============================
H = 32
Ny = 2 * H + 1
Nx = 1024
Nt = 5000
tempos_saida = [Nt//4, Nt//2, 3*Nt//4, Nt]
posicoes_x = [Nx//4, Nx//2, 3*Nx//4, Nx-1]

# ===============================
# Parâmetros LBM conhecidos: Reynolds e tau
# ===============================
Re = 100.0       # Reynolds desejado
tau = 0.6        # tempo de relaxação BGK
rho0 = 1.0       # densidade

# viscosidade a partir de tau
nu = (tau - 0.5) / 3

# velocidade de entrada a partir de Re
H_phys = H      # metade do canal em unidades lattice
u_max = Re * nu / (2 * H_phys)
u_entrada = u_max / 1.5  # relaciona velocidade central com entrada (Poiseuille 2D)

# omega
omega = 1.0 / tau

print(f"nu = {nu:.6f}, u_entrada = {u_entrada:.6f}, u_max = {u_max:.6f}, tau = {tau:.2f}")

# ===============================
# Constantes D2Q9
# ===============================
w = np.array([4/9] + [1/9]*4 + [1/36]*4)
c = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1],
              [1,1], [-1,1], [-1,-1], [1,-1]]).astype(int)

# função de equilíbrio
def feq(rho, ux, uy):
    if rho.ndim == 1:
        feq_local = np.zeros((9, Ny))
        u2 = ux**2 + uy**2
        for i in range(9):
            cu = 3 * (c[i,0]*ux + c[i,1]*uy)
            feq_local[i] = w[i] * rho * (1 + cu + 0.5*cu**2 - 1.5*u2)
    else:
        feq_local = np.zeros((9, Ny, Nx))
        u2 = ux**2 + uy**2
        for i in range(9):
            cu = 3 * (c[i,0]*ux + c[i,1]*uy)
            feq_local[i] = w[i] * rho * (1 + cu + 0.5*cu**2 - 1.5*u2)
    return feq_local

# ===============================
# Inicialização
# ===============================
rho = np.ones((Ny, Nx))
ux = np.zeros((Ny, Nx))
uy = np.zeros((Ny, Nx))
f = feq(rho, ux, uy)

is_solid = np.zeros((Ny, Nx), dtype=bool)
is_solid[0, :] = True
is_solid[-1, :] = True
opposite = [0,3,4,1,2,7,8,5,6]

perfis = {x: {} for x in posicoes_x}

# ===============================
# Loop principal com Zou–He
# ===============================
for t in range(1, Nt+1):
    # --- Cálculo dos macroscópicos ---
    rho = f.sum(axis=0)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    for i in range(9):
        ux += f[i] * c[i,0]
        uy += f[i] * c[i,1]
    ux /= rho
    uy /= rho

    # --- Condição de entrada Zou–He em x=0 ---
    ux[:,0] = u_entrada
    uy[:,0] = 0.0
    rho[:,0] = (f[0,:,0] + f[2,:,0] + f[4,:,0] +
                2*(f[3,:,0] + f[6,:,0] + f[7,:,0])) / (1 - ux[:,0])
    
    # reconstrução das populações desconhecidas (1,5,8)
    f[1,:,0] = f[3,:,0] + 2/3 * rho[:,0] * ux[:,0]
    f[5,:,0] = f[7,:,0] + 0.5*(f[4,:,0]-f[2,:,0]) + 1/6 * rho[:,0]*ux[:,0] + 0.5 * rho[:,0]*uy[:,0]
    f[8,:,0] = f[6,:,0] + 0.5*(f[2,:,0]-f[4,:,0]) + 1/6 * rho[:,0]*ux[:,0] - 0.5 * rho[:,0]*uy[:,0]

    # --- Colisão ---
    feq_local = feq(rho, ux, uy)
    f = f - omega * (f - feq_local)

    # --- Streaming ---
    for i in range(9):
        f[i] = np.roll(np.roll(f[i], c[i,1], axis=0), c[i,0], axis=1)

    # --- Bounce-back nas paredes ---
    for i in range(9):
        f[i][is_solid] = f[opposite[i]][is_solid]

    # --- Saída (Neumann simples) ---
    f[:, :, -1] = f[:, :, -2]

    # --- Armazenar perfis ---
    if t in tempos_saida:
        rho = f.sum(axis=0)
        ux = np.zeros_like(rho)
        uy = np.zeros_like(rho)
        for i in range(9):
            ux += f[i] * c[i,0]
            uy += f[i] * c[i,1]
        ux /= rho
        uy /= rho
        for x in posicoes_x:
            perfis[x][t] = ux[:, x].copy()

# ===============================
# Cálculo do deltaP implícito
# ===============================
u_max = u_entrada * 1.5  # relación para Poiseuille 2D (entrada central)
deltaP = 2 * nu * rho0 * u_max / H**2
print(f"DeltaP implícito = {deltaP:.6f}")

# ===============================
# Perfil teórico de Poiseuille a partir de deltaP
# ===============================
y = np.arange(Ny)
y_central = y - H
u_teorico = (deltaP / (2 * nu * rho0)) * (H**2 - y_central**2)

# ===============================
# Plot de evolução dos perfis com solução teórica
# ===============================
for x in posicoes_x:
    plt.figure(figsize=(6,5))
    for t in tempos_saida:
        if t in perfis[x]:
            plt.plot(perfis[x][t], y, label=f"t={t}")
    plt.plot(u_teorico, y, 'k--', label="Solução teórica")
    plt.xlabel("u_x")
    plt.ylabel("y")
    plt.title(f"Evolução do perfil em x={x}")
    plt.legend()
    plt.grid(True)
    plt.show()

# ===============================
# Heatmap da magnitude da velocidade no último tempo
# ===============================
vel_mag = np.sqrt(ux**2 + uy**2)

plt.figure(figsize=(12,4))
plt.imshow(vel_mag, origin='lower', cmap='viridis', 
           extent=[0, Nx-1, 0, Ny-1], aspect='auto')
plt.colorbar(label='|u|')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Magnitude da velocidade |u| em t = Nt')
plt.show()
