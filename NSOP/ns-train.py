import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# 1. Load and process Data
print("Loading dataset...")
df = pd.read_csv('nvidia_stock_data_1999_2026.csv')

df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df = df.dropna(subset=['date', 'close'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Loaded {len(df)} valid price rows.")

today_date = pd.to_datetime('today').normalize()
current_row = df[df['date'] <= today_date]
if not current_row.empty:
    curr_price = current_row.iloc[-1]['close']
    curr_dt = current_row.iloc[-1]['date']
else:
    curr_price = df.iloc[-1]['close']
    curr_dt = df.iloc[-1]['date']

df['return'] = df['close'].pct_change()
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['return'])

data = df['return'].values.reshape(-1, 1)
dates = df['date'].values
prices = df['close'].values

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data)

# 2. Create Sequences
def create_sequences(dataset, window_size):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i-window_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)

if len(X) == 0:
    raise ValueError("Error: Sequence array is empty.")

y_dates = dates[window_size:]
y_prices = prices[window_size:]

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.85)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
test_dates = y_dates[train_size:]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 3. Model Setup
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=False)

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1, num_layers=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.linear(out[:, -1, :])

model = StockLSTM()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Training
print("Training model...")
model.train()
for epoch in range(50):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {epoch_loss/len(train_loader):.6f}")

# 5. Forecast
model.eval()
last_known_window = X[-1].copy()
target_end_date = pd.to_datetime('2030-12-31')

future_return_preds = []
future_dates = []

current_date = pd.to_datetime(y_dates[-1])
last_price = y_prices[-1]
future_prices = []

with torch.no_grad():
    while current_date < target_end_date:
        current_date += timedelta(days=1)
        if current_date.weekday() >= 5:
            continue
            
        x_tensor = torch.tensor(last_known_window, dtype=torch.float32).unsqueeze(0)
        pred_scaled = model(x_tensor).item()
        
        pred_return = scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]
        last_price = last_price * (1 + pred_return)
        
        future_prices.append(last_price)
        future_dates.append(np.datetime64(current_date, 'D'))
        
        last_known_window = np.vstack((last_known_window[1:], [[pred_scaled]]))

test_return_preds = scaler.inverse_transform(model(X_test_tensor).detach().numpy())
test_reconstructed_prices = []
curr_p = y_prices[train_size - 1]
for r in test_return_preds:
    curr_p = curr_p * (1 + r[0])
    test_reconstructed_prices.append(curr_p)

all_pred_dates = np.concatenate([test_dates.astype('datetime64[D]'), np.array(future_dates)])
all_pred_values = np.concatenate([np.array(test_reconstructed_prices), np.array(future_prices)])

# 6. Plotting
PHP_EXCHANGE_RATE = 60

plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(15, 8), facecolor='#090d16')
ax1.set_facecolor('#0d1322')

recent_mask = df['date'] >= '2022-01-01'
hist_dates = df.loc[recent_mask, 'date'].values.astype('datetime64[D]')
hist_prices = df.loc[recent_mask, 'close'].values

ax1.plot(hist_dates, hist_prices, label='Actual Historical Price (USD)', color='#38bdf8', linewidth=2.5, zorder=3)
ax1.fill_between(hist_dates, hist_prices, color='#38bdf8', alpha=0.1, zorder=2)

ax1.plot(all_pred_dates, all_pred_values, label='LSTM Return-Based Outlook to 2030 (USD)', color='#f43f5e', linestyle='--', linewidth=2, zorder=3)
ax1.fill_between(all_pred_dates, all_pred_values, color='#f43f5e', alpha=0.08, zorder=2)

ax1.axhline(y=curr_price, color='#22c55e', linestyle='-', linewidth=1.5, label=f'Current Price (${curr_price:.2f})', zorder=4)

# Custom USD ticks and limits
usd_ticks = [0, 10, 20, 25, 50]
ax1.set_yticks(usd_ticks)
ax1.set_ylim(0, 50)

ax1.set_title('NVIDIA - Long-Term Stock Outlook until 2030', fontsize=16, pad=20, fontweight='bold', color='#f8fafc')
ax1.set_xlabel('Timeline (Years)', fontsize=12, labelpad=12, color='#cbd5e1', fontweight='semibold')
ax1.set_ylabel('Price (USD)', fontsize=12, labelpad=12, color='#38bdf8', fontweight='semibold')
ax1.tick_params(axis='y', labelcolor='#38bdf8', labelsize=10)
ax1.tick_params(axis='x', labelcolor='#cbd5e1', labelsize=10)

ax1.grid(True, color='#1e293b', linestyle='--', linewidth=0.8, alpha=0.8)
ax1.xaxis.set_major_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Secondary Y-axis for PHP matching USD ticks exactly
ax2 = ax1.twinx()
ax2.set_facecolor('none')
php_ticks = [t * PHP_EXCHANGE_RATE for t in usd_ticks]
ax2.set_yticks(php_ticks)
ax2.set_ylim(0, 50 * PHP_EXCHANGE_RATE)
ax2.set_ylabel('Price (PHP)', fontsize=12, labelpad=12, color='#f43f5e', fontweight='semibold')
ax2.tick_params(axis='y', labelcolor='#f43f5e', labelsize=10)

for spine in ['top', 'bottom', 'left', 'right']:
    ax1.spines[spine].set_color('#334155')
    ax2.spines[spine].set_color('#334155')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, facecolor='#111827', edgecolor='#334155', fontsize=11, framealpha=0.9, shadow=True)

plt.tight_layout()
plt.show()