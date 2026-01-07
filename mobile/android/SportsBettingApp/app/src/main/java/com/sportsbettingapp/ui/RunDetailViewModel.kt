package com.sportsbettingapp.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sportsbettingapp.data.model.Prediction
import com.sportsbettingapp.data.repository.ApiRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class RunDetailViewModel(
    private val repository: ApiRepository = ApiRepository.create()
) : ViewModel() {
    private val _predictionsState = MutableStateFlow<UiState<List<Prediction>>>(UiState.Loading)
    val predictionsState: StateFlow<UiState<List<Prediction>>> = _predictionsState

    fun loadPredictions(runId: String) {
        _predictionsState.value = UiState.Loading
        viewModelScope.launch {
            try {
                val predictions = repository.getPredictions(runId)
                _predictionsState.value = UiState.Success(predictions)
            } catch (exception: Exception) {
                _predictionsState.value = UiState.Error(exception.localizedMessage ?: "Failed to load predictions")
            }
        }
    }
}
