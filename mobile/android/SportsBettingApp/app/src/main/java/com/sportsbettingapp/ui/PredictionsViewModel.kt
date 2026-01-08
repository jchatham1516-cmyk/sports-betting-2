package com.sportsbettingapp.ui

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sportsbettingapp.data.model.PredictionsResponseDto
import com.sportsbettingapp.data.model.RunStatusResponseDto
import com.sportsbettingapp.data.repository.ApiRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class PredictionsViewModel(
    private val savedStateHandle: SavedStateHandle,
    private val repository: ApiRepository = ApiRepository.create()
) : ViewModel() {
    private val _predictionsState = MutableStateFlow<UiState<PredictionsResponseDto>>(UiState.Idle)
    val predictionsState: StateFlow<UiState<PredictionsResponseDto>> = _predictionsState

    private val _statusState = MutableStateFlow<UiState<RunStatusResponseDto>>(UiState.Idle)
    val statusState: StateFlow<UiState<RunStatusResponseDto>> = _statusState

    private var pollingJob: Job? = null

    fun startPolling(runId: String) {
        if (pollingJob?.isActive == true && savedStateHandle[RUN_ID_KEY] == runId) {
            return
        }
        savedStateHandle[RUN_ID_KEY] = runId
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            while (true) {
                try {
                    val status = withContext(Dispatchers.IO) {
                        repository.getRunStatus(runId)
                    }
                    _statusState.value = UiState.Success(status)
                    if (status.status == "succeeded") {
                        loadPredictions(runId)
                        break
                    }
                    if (status.status == "failed") {
                        break
                    }
                } catch (exception: Exception) {
                    _statusState.value = UiState.Error(exception.localizedMessage ?: "Failed to load status")
                    break
                }
                delay(1_500L)
            }
        }
    }

    private fun loadPredictions(runId: String) {
        _predictionsState.value = UiState.Loading
        viewModelScope.launch {
            try {
                val predictions = withContext(Dispatchers.IO) {
                    repository.getPredictions(runId)
                }
                _predictionsState.value = UiState.Success(predictions)
            } catch (e: Exception) {
                _predictionsState.value = UiState.Error(e.message ?: "Failed to load predictions")
            }
        }
    }

    companion object {
        private const val RUN_ID_KEY = "run_id"
    }
}
