package com.sportsbettingapp.ui.screens

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
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import com.sportsbettingapp.data.model.Bet
import com.sportsbettingapp.ui.BetsViewModel
import com.sportsbettingapp.ui.UiState
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BetsScreen(
    viewModel: BetsViewModel,
    contentPadding: PaddingValues
) {
    val betsState by viewModel.betsState.collectAsState()
    var sportExpanded by remember { mutableStateOf(false) }
    var selectedSport by remember { mutableStateOf("nba") }
    var selectedDate by remember { mutableStateOf(LocalDate.now()) }
    var showDatePicker by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(16.dp)
    ) {
        Text(text = "Tracked Bets", style = MaterialTheme.typography.headlineSmall)
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

        Spacer(modifier = Modifier.height(16.dp))
        Button(
            onClick = {
                viewModel.loadBets(
                    date = selectedDate.format(DateTimeFormatter.ISO_DATE),
                    sport = selectedSport
                )
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Fetch Bets")
        }

        Spacer(modifier = Modifier.height(16.dp))
        when (val state = betsState) {
            UiState.Idle -> Text("Select a date and sport to load bets.")
            UiState.Loading -> {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    CircularProgressIndicator()
                }
            }
            is UiState.Error -> Text(text = state.message, color = MaterialTheme.colorScheme.error)
            is UiState.Success -> BetsList(bets = state.data)
        }
    }
}

@Composable
private fun BetsList(bets: List<Bet>) {
    if (bets.isEmpty()) {
        Text("No bets found.")
        return
    }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        items(bets) { bet ->
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(12.dp)
            ) {
                Text(text = bet.matchup, style = MaterialTheme.typography.titleMedium)
                Text(text = "Market: ${bet.market ?: "-"}")
                Text(text = "Pick: ${bet.pick ?: "-"}")
                Text(text = "Units: ${bet.units ?: "-"}")
                Text(text = "Result: ${bet.result ?: "-"}")
            }
        }
    }
}
