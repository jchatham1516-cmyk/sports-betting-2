package com.sportsbettingapp.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sportsbettingapp.data.repository.ApiRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class PredictionsViewModel : ViewModel() {
    private val repository = ApiRepository.create()

    private val _predictionsState = MutableStateFlow<UiState<List<com.sportsbettingapp.data.model.Prediction>>>(UiState.Idle)
    val predictionsState: StateFlow<UiState<List<com.sportsbettingapp.data.model.Prediction>>> = _predictionsState

    fun loadPredictions(runId: String) {
        _predictionsState.value = UiState.Loading
        viewModelScope.launch {
            try {
                val predictions = repository.getPredictions(runId)
                _predictionsState.value = UiState.Success(predictions)
            } catch (e: Exception) {
                _predictionsState.value = UiState.Error(e.message ?: "Failed to load predictions")
            }
        }
    }
}
