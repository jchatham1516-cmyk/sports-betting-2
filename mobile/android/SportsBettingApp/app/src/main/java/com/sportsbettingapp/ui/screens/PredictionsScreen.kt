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
import com.sportsbettingapp.data.model.Prediction
import com.sportsbettingapp.ui.PredictionsViewModel
import com.sportsbettingapp.ui.UiState

@Composable
fun PredictionsScreen(
    runId: String,
    viewModel: PredictionsViewModel,
    contentPadding: PaddingValues
) {
    val predictionsState by viewModel.predictionsState.collectAsState()
    val context = LocalContext.current

    LaunchedEffect(runId) {
        viewModel.loadPredictions(runId)
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

        when (val state = predictionsState) {
            UiState.Loading -> {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    CircularProgressIndicator()
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
private fun PredictionsList(predictions: List<Prediction>) {
    if (predictions.isEmpty()) {
        Text("No predictions found.")
        return
    }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        items(predictions) { prediction ->
            PredictionRow(prediction = prediction)
        }
    }
}

@Composable
private fun PredictionRow(prediction: Prediction) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(12.dp)
    ) {
        Text(text = "${prediction.away} @ ${prediction.home}", style = MaterialTheme.typography.titleMedium)
        Text(text = "Recommendation: ${prediction.primaryRecommendation ?: "-"}")
        Text(text = "Confidence: ${prediction.confidence ?: "-"}")
        Text(text = "Value tier: ${prediction.valueTier ?: "-"}")
        prediction.edges?.let { edges ->
            val formatted = edges.entries.joinToString { "${it.key}: ${it.value}" }
            Text(text = "Edges: $formatted")
        }
    }
}
