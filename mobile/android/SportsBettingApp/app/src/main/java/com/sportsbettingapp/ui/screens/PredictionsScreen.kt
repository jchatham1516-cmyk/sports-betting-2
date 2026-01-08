package com.sportsbettingapp.ui.screens

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.sportsbettingapp.BuildConfig
import com.sportsbettingapp.data.model.PredictionItemDto
import com.sportsbettingapp.data.model.PredictionsResponseDto
import com.sportsbettingapp.ui.PredictionsViewModel
import com.sportsbettingapp.ui.UiState

@Composable
fun PredictionsScreen(
    runId: String,
    viewModel: PredictionsViewModel,
    contentPadding: PaddingValues
) {
    val predictionsState by viewModel.predictionsState.collectAsState()
    val statusState by viewModel.statusState.collectAsState()
    val context = LocalContext.current

    LaunchedEffect(runId) {
        viewModel.startPolling(runId)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(16.dp)
    ) {
        Text(text = "Predictions", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(8.dp))
        Text(text = "Run ID: $runId")
        Spacer(modifier = Modifier.height(16.dp))
        Button(
            onClick = {
                val url = "${BuildConfig.BASE_URL}/api/runs/$runId/download/predictions.csv"
                val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
                context.startActivity(intent)
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Open CSV")
        }
        Spacer(modifier = Modifier.height(16.dp))

        when (val status = statusState) {
            is UiState.Success -> {
                val progress = status.data.progress.coerceIn(0, 100)
                LinearProgressIndicator(
                    progress = progress / 100f,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text("${progress}%")
                Spacer(modifier = Modifier.height(4.dp))
                Text(status.data.message)
                status.data.error?.let { error ->
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(text = error, color = MaterialTheme.colorScheme.error)
                }
                Spacer(modifier = Modifier.height(16.dp))
            }
            is UiState.Error -> {
                Text(text = status.message, color = MaterialTheme.colorScheme.error)
                Spacer(modifier = Modifier.height(16.dp))
            }
            UiState.Loading -> Unit
            UiState.Idle -> Unit
        }

        when (val state = predictionsState) {
            UiState.Loading -> {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("Loading predictions...")
                }
            }
            is UiState.Error -> {
                Text(text = state.message, color = MaterialTheme.colorScheme.error)
            }
            is UiState.Success -> {
                PredictionsList(predictions = state.data)
            }
            UiState.Idle -> Unit
        }
    }
}

@Composable
private fun PredictionsList(predictions: PredictionsResponseDto) {
    if (predictions.items.isEmpty()) {
        Text("No predictions found.")
        return
    }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        items(predictions.items) { prediction ->
            PredictionRow(prediction = prediction)
        }
    }
}

@Composable
private fun PredictionRow(prediction: PredictionItemDto) {
    val pickText = prediction.pick ?: prediction.primaryRecommendation ?: "-"
    val matchup = when {
        !prediction.away.isNullOrBlank() || !prediction.home.isNullOrBlank() ->
            "${prediction.away ?: ""} @ ${prediction.home ?: ""}".trim()
        else -> "Matchup unavailable"
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(12.dp)
    ) {
        Text(text = matchup, style = MaterialTheme.typography.titleMedium)
        prediction.date?.let { Text(text = "Date: $it") }
        Text(text = "Pick: $pickText")
        prediction.confidence?.let { Text(text = "Confidence: $it") }
        prediction.odds?.let { Text(text = "Odds: $it") }
        prediction.edge?.let { Text(text = "Edge: $it") }
        prediction.price?.let { Text(text = "Price: $it") }
        prediction.units?.let { Text(text = "Units: $it") }
    }
}
