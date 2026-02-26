import torch
import torch.nn as nn

class LSTMDropoutPredictor(nn.Module):
    def __init__(self, sequence_features, static_features, hidden_dim=64, num_layers=2, dropout=0.3):
        super(LSTMDropoutPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # The LSTM layer takes the chronological sequence data (e.g., clicks, hesitation per week)
        self.lstm = nn.LSTM(
            input_size=sequence_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer to merge LSTM output and static features (demographics)
        self.fc_merge = nn.Linear(hidden_dim + static_features, hidden_dim // 2)
        
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        
        # Final classification head
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_seq, x_static):
        """
        x_seq: (Batch, Sequence_Length, Seq_Features)
        x_static: (Batch, Static_Features)
        """
        # LSTM output
        # out: (batch, seq_len, hidden_size)
        # hn: (num_layers, batch, hidden_size) - hidden state of last time step
        lstm_out, (hn, cn) = self.lstm(x_seq)
        
        # We take the hidden state of the final layer of the LSTM
        # Shape: (Batch, Hidden_Dim)
        last_hidden_state = hn[-1, :, :]
        
        # Concatenate sequential embedding with static demographic embedding
        combined = torch.cat((last_hidden_state, x_static), dim=1)
        
        # Pass through dense layers
        x = self.fc_merge(combined)
        x = self.relu(x)
        x = self.dropout_layer(x)
        
        # Logits and Probability
        logits = self.classifier(x)
        probs = self.sigmoid(logits)
        
        return probs

def train_lstm(model, dataloader_train, dataloader_val, epochs=50, lr=1e-3, device='cpu'):
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_seq, batch_static, batch_y in dataloader_train:
            batch_seq, batch_static, batch_y = batch_seq.to(device), batch_static.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # Ensure target shape matches output shape: (Batch, 1)
            outputs = model(batch_seq, batch_static)
            batch_y = batch_y.unsqueeze(1).float()
            
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_seq, batch_static, batch_y in dataloader_val:
                batch_seq, batch_static, batch_y = batch_seq.to(device), batch_static.to(device), batch_y.to(device)
                
                outputs = model(batch_seq, batch_static)
                batch_y = batch_y.unsqueeze(1).float()
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(dataloader_train)
        avg_val_loss = val_loss / len(dataloader_val)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save best weights here
            
    return model
