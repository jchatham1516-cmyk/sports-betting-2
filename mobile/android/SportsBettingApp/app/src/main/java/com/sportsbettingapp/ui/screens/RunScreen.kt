package com.sportsbettingapp.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import com.sportsbettingapp.ui.RunViewModel
import com.sportsbettingapp.ui.UiState
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RunScreen(
    viewModel: RunViewModel,
    onRunSuccess: (String) -> Unit,
    contentPadding: PaddingValues
) {
    val runState by viewModel.runState.collectAsState()
    val statusState by viewModel.statusState.collectAsState()
    val currentRunId by viewModel.currentRunId.collectAsState()
    var sportExpanded by remember { mutableStateOf(false) }
    var selectedSport by remember { mutableStateOf("nba") }
    var selectedDate by remember { mutableStateOf(LocalDate.now()) }
    var showDatePicker by remember { mutableStateOf(false) }

    LaunchedEffect(runState) {
        if (runState is UiState.Success) {
            onRunSuccess((runState as UiState.Success<String>).data)
            viewModel.reset()
        }
    }

    LaunchedEffect(currentRunId) {
        currentRunId?.let { runId ->
            viewModel.startStatusPolling(runId)
        }
    }

    val isRunActive = (statusState as? UiState.Success)?.data?.status in listOf("queued", "running")

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(16.dp),
        verticalArrangement = Arrangement.Top
    ) {
        Text(text = "Run Model", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(16.dp))

        ExposedDropdownMenuBox(
            expanded = sportExpanded,
            onExpandedChange = { sportExpanded = it }
        ) {
            OutlinedTextField(
                value = TextFieldValue(selectedSport),
                onValueChange = {},
                modifier = Modifier
                    .menuAnchor()
                    .fillMaxWidth(),
                readOnly = true,
                label = { Text("Sport") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = sportExpanded) }
            )
            ExposedDropdownMenu(
                expanded = sportExpanded,
                onDismissRequest = { sportExpanded = false }
            ) {
                listOf("nba", "nfl", "nhl").forEach { option ->
                    androidx.compose.material3.DropdownMenuItem(
                        text = { Text(option) },
                        onClick = {
                            selectedSport = option
                            sportExpanded = false
                        }
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        Button(
            onClick = { showDatePicker = true },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Date: ${selectedDate.format(DateTimeFormatter.ISO_DATE)}")
        }

        if (showDatePicker) {
            val datePickerState = androidx.compose.material3.rememberDatePickerState(
                initialSelectedDateMillis = selectedDate.atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli()
            )
            DatePickerDialog(
                onDismissRequest = { showDatePicker = false },
                confirmButton = {
                    TextButton(
                        onClick = {
                            val millis = datePickerState.selectedDateMillis
                            if (millis != null) {
                                selectedDate = Instant.ofEpochMilli(millis)
                                    .atZone(ZoneId.systemDefault())
                                    .toLocalDate()
                            }
                            showDatePicker = false
                        }
                    ) {
                        Text("OK")
                    }
                },
                dismissButton = {
                    TextButton(onClick = { showDatePicker = false }) {
                        Text("Cancel")
                    }
                }
            ) {
                DatePicker(state = datePickerState)
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
        Button(
            onClick = {
                viewModel.runModel(
                    sport = selectedSport,
                    gameDate = selectedDate.format(DateTimeFormatter.ISO_DATE)
                )
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = !isRunActive && runState !is UiState.Loading
        ) {
            Text(if (isRunActive) "Run In Progress" else "Run Model")
        }

        Spacer(modifier = Modifier.height(16.dp))
        when (val state = runState) {
            UiState.Idle -> Unit
            UiState.Loading -> {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("Running model...")
                }
            }
            is UiState.Error -> {
                Text(
                    text = state.message,
                    color = MaterialTheme.colorScheme.error
                )
            }
            is UiState.Success -> Unit
        }

        when (val status = statusState) {
            is UiState.Success -> {
                val progress = status.data.progressPercent.coerceIn(0, 100)
                Spacer(modifier = Modifier.height(16.dp))
                LinearProgressIndicator(
                    progress = progress / 100f,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text("${progress}%")
                Spacer(modifier = Modifier.height(4.dp))
                Text(status.data.message ?: "Working...")
                status.data.error?.let { error ->
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(text = error, color = MaterialTheme.colorScheme.error)
                }
            }
            is UiState.Error -> {
                Spacer(modifier = Modifier.height(16.dp))
                Text(text = status.message, color = MaterialTheme.colorScheme.error)
            }
            UiState.Loading -> Unit
            UiState.Idle -> Unit
        }

        if (isRunActive && currentRunId != null) {
            Spacer(modifier = Modifier.height(16.dp))
            Button(
                onClick = { onRunSuccess(currentRunId ?: "") },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Resume run")
            }
        }
    }
}
