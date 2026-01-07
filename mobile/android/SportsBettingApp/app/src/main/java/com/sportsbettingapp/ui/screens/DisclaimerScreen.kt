package com.sportsbettingapp.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun DisclaimerScreen(contentPadding: PaddingValues) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(16.dp)
    ) {
        Text(text = "Disclaimer", style = MaterialTheme.typography.headlineSmall)
        Text(
            text = "Predictions and recommendations are informational only and do not constitute financial advice. " +
                "All betting involves risk. Please wager responsibly.",
            modifier = Modifier.padding(top = 12.dp)
        )
    }
}
